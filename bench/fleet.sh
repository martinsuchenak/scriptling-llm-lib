#!/usr/bin/env bash
#
# fleet.sh — cross-compile the `infer` binary, push it (and the selected models)
# to remote hosts, run benchmarks / CPU profiles, and collect results.
#
# Remotes need NOTHING but ssh access — the binary is cross-compiled here and
# the models are copied from a local source dir. GOAMD64 is left at v1 so one
# linux/amd64 build runs on every x86 box (incl. the no-AVX Xeon X5675); the
# library selects SIMD kernels at runtime via CPUID.
#
# Supports linux, darwin and windows targets. Windows hosts are driven over
# OpenSSH assuming the default cmd.exe shell; write their paths in the config
# with forward slashes (e.g. C:/models). Unix paths may use ~ but not spaces.
#
# Usage:
#   ./bench/fleet.sh check [host|all]              connectivity + CPU info
#   ./bench/fleet.sh deploy <host|all>             mkdir + copy binary + models
#   ./bench/fleet.sh run <host> <model> [args...]  one generation, raw output
#   ./bench/fleet.sh bench [host|all]              standard t/s matrix table
#   ./bench/fleet.sh profile <host> <model> [toks] CPU profile -> pprof -top
#   ./bench/fleet.sh models                         list models MODELS=all resolves to
#
# <model> is a .gguf basename (copied from the local source dir and resolved
# against the host's model_dir) or an absolute path already on the remote.
#
# Pick which models deploy/bench act on with MODELS. Set MODELS=all to use every
# .gguf in the local source dir (the easy way to test all models, k-quants
# included); otherwise it's a space-separated basename list. Examples:
#   MODELS=all ./bench/fleet.sh bench m2max
#   MODELS="SmolLM2-1.7B-Instruct-Q4_K_M.gguf SmolLM2-1.7B-Instruct-Q8_0.gguf" \
#       ./bench/fleet.sh bench 9900x
#
# Env overrides: FLEET_CONF, FLEET_PROMPT, FLEET_TOKENS, SSH_OPTS, MODELS,
#   FLEET_MODELS_SRC (local dir holding the .gguf files; default <repo>/models),
#   FORCE=1 (re-copy models even if present).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONF="${FLEET_CONF:-$HERE/hosts.conf}"
# Fall back to the committed example if no local hosts.conf exists yet.
[ -f "$CONF" ] || CONF="$HERE/hosts.conf.example"
BINDIR="$HERE/bin"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8}"
PROMPT="${FLEET_PROMPT:-Write a long, detailed story about a dog and its adventures across the world.}"
TOKENS="${FLEET_TOKENS:-120}"
MODELS="${MODELS:-SmolLM2-135M-Instruct-Q8_0.gguf SmolLM2-360M-Instruct-Q8_0.gguf SmolLM2-1.7B-Instruct-Q8_0.gguf SmolLM2-1.7B-Instruct-Q4_0.gguf}"
LOCAL_MODELS="${FLEET_MODELS_SRC:-$ROOT/models}"

die() { echo "fleet: $*" >&2; exit 1; }

# resolve_models echoes the model list to act on. MODELS=all (or "*") expands to
# every .gguf basename in LOCAL_MODELS — the easy way to test every model,
# including the k-quant ones — sorted for a stable table order. Otherwise the
# explicit MODELS list (or its default) is used verbatim.
resolve_models() {
	case "${MODELS:-}" in
		all|'*'|'')
			local f found=
			for f in "$LOCAL_MODELS"/*.gguf; do
				[ -e "$f" ] || continue
				found="$found $(basename "$f")"
			done
			[ -n "$found" ] || die "no .gguf files in $LOCAL_MODELS (set FLEET_MODELS_SRC or MODELS)"
			printf '%s\n' $found | sort | tr '\n' ' '
			;;
		*) printf '%s' "$MODELS" ;;
	esac
}

load_host() {
	local want="$1" line
	line="$(awk -v n="$want" '!/^[[:space:]]*#/ && NF>=6 && $1==n {print; exit}' "$CONF")"
	[ -n "$line" ] || die "host '$want' not found in $CONF"
	# shellcheck disable=SC2086
	set -- $line
	NAME="$1"; SSH="$2"; GOOS="$3"; GOARCH="$4"; RDIR="${5%/}"; MDIR="${6%/}"
}

all_hosts() { awk '!/^[[:space:]]*#/ && NF>=6 {print $1}' "$CONF"; }
is_local()  { [ "$SSH" = "local" ]; }
is_win()    { [ "$GOOS" = "windows" ]; }
bin_name()  { is_win && echo "infer.exe" || echo "infer"; }
is_abs()    { case "$1" in /*|[A-Za-z]:*) return 0;; *) return 1;; esac; }

build_for() { # goos goarch -> echoes local binary path
	local out="$BINDIR/${1}_${2}/infer"; is_win && out="$out.exe"
	mkdir -p "$(dirname "$out")"
	( cd "$ROOT" && GOOS="$1" GOARCH="$2" go build -o "$out" ./examples/infer )
	echo "$out"
}

rsh() { if is_local; then bash -c "$1"; else ssh $SSH_OPTS "$SSH" "$1"; fi; }
q()       { if is_win; then printf '"%s"' "$1"; else printf "'%s'" "$1"; fi; }  # quote prompt
winpath() { printf '%s' "$1" | tr '/' '\\'; }
remote_bin() { if is_local; then build_for "$GOOS" "$GOARCH"; else echo "$RDIR/$(bin_name)"; fi; }
resolve_model() { if is_abs "$1"; then echo "$1"; else echo "$MDIR/$1"; fi; }

ensure_dirs() {
	if is_local; then mkdir -p "$RDIR" "$MDIR"; return; fi
	if is_win; then
		rsh "if not exist \"$(winpath "$RDIR")\" mkdir \"$(winpath "$RDIR")\"" || die "[$NAME] ssh failed"
		rsh "if not exist \"$(winpath "$MDIR")\" mkdir \"$(winpath "$MDIR")\"" || true
	else
		rsh "mkdir -p $RDIR $MDIR" || die "[$NAME] ssh failed"   # unquoted: ~ expands
	fi
}

push_binary() {
	local bin; bin="$(build_for "$GOOS" "$GOARCH")"
	is_local && return
	scp $SSH_OPTS -q "$bin" "$SSH:$RDIR/$(bin_name)" || die "[$NAME] binary scp failed"
	is_win || rsh "chmod +x $RDIR/$(bin_name)"
}

model_present() {
	if is_local; then [ -e "$MDIR/$1" ]; return; fi
	local out
	if is_win; then out="$(rsh "if exist \"$(winpath "$MDIR/$1")\" echo Y" 2>/dev/null || true)"
	else out="$(rsh "test -e $MDIR/$1 && echo Y" 2>/dev/null || true)"; fi
	[ -n "$out" ]
}

ensure_model() { # basename — copy from LOCAL_MODELS if not already on the host
	local b="$1" src="$LOCAL_MODELS/$1"
	is_abs "$b" && return 0   # absolute path: assume already on the remote
	if [ -z "${FORCE:-}" ] && model_present "$b"; then return 0; fi
	[ -f "$src" ] || die "[$NAME] local model not found: $src (set FLEET_MODELS_SRC)"
	echo "[$NAME] copying $b ($(du -h "$src" | cut -f1)) -> $MDIR ..." >&2
	if is_local; then cp "$src" "$MDIR/$b"; else scp $SSH_OPTS "$src" "$SSH:$MDIR/$b"; fi
}

infer_cmd() { # model tokens prof [extra...]
	local model="$1" toks="$2" prof="${3:-}"; shift 3 || true
	local bin c
	bin="$(remote_bin)"
	if is_win; then
		c="\"$(winpath "$bin")\" -model \"$(winpath "$model")\" -prompt $(q "$PROMPT") -tokens $toks -strategy greedy"
		[ -n "$prof" ] && c="$c -prof \"$(winpath "$prof")\""
	else
		c="$bin -model $model -prompt $(q "$PROMPT") -tokens $toks -strategy greedy"   # paths unquoted: ~ expands
		[ -n "$prof" ] && c="$c -prof $prof"
	fi
	printf '%s %s' "$c" "$*"
}

deploy() {
	load_host "$1"; ensure_dirs; push_binary
	local m; for m in $(resolve_models); do ensure_model "$m"; done
	echo "[$NAME] ready (bin in $RDIR, models in $MDIR)"
}

run_host() {
	load_host "$1"; shift
	local marg="$1"; shift
	ensure_dirs; push_binary; ensure_model "$marg"
	rsh "$(infer_cmd "$(resolve_model "$marg")" "$TOKENS" "" "$@")"
}

bench_host() {
	load_host "$1"; ensure_dirs; push_binary
	printf '== %-9s (%s/%s, %s) ==\n' "$NAME" "$GOOS" "$GOARCH" "$SSH"
	printf '%-38s %10s %10s\n' "model" "prefill" "decode"
	local m out pre dec
	for m in $(resolve_models); do
		ensure_model "$m" || { printf '%-38s %10s\n' "$m" "NO-MODEL"; continue; }
		if out="$(rsh "$(infer_cmd "$(resolve_model "$m")" "$TOKENS" "")" 2>&1)"; then
			pre="$(printf '%s\n' "$out" | awk '/prefill/{print $(NF-1)" "$NF}')"
			dec="$(printf '%s\n' "$out" | awk '/decode/{print $(NF-1)" "$NF}')"
			printf '%-38s %10s %10s\n' "$m" "${pre:-?}" "${dec:-?}"
		else
			printf '%-38s %10s\n' "$m" "ERR"
		fi
	done
	echo
}

profile_host() {
	load_host "$1"
	local model toks rprof lprof
	model="$2"; toks="${3:-150}"
	ensure_dirs; push_binary; ensure_model "$model"
	rprof="$RDIR/cpu.prof"; is_local && rprof="/tmp/fleet-$NAME.prof"
	echo "[$NAME] profiling $model ($toks tokens)..."
	rsh "$(infer_cmd "$(resolve_model "$model")" "$toks" "$rprof")" >/dev/null 2>&1 || true
	lprof="$BINDIR/${NAME}.prof"
	if is_local; then cp "$rprof" "$lprof"; else scp $SSH_OPTS -q "$SSH:$rprof" "$lprof" || die "[$NAME] could not fetch profile"; fi
	echo "[$NAME] top functions:"
	go tool pprof -top -nodecount=20 "$(build_for "$GOOS" "$GOARCH")" "$lprof" 2>/dev/null | tail -24
}

check_host() {
	load_host "$1"
	printf '== %-9s (%s, %s/%s) ==\n' "$NAME" "$SSH" "$GOOS" "$GOARCH"
	if rsh 'uname -sm 2>/dev/null || ver' >/tmp/.fleetchk 2>/dev/null; then
		cat /tmp/.fleetchk
		rsh 'lscpu 2>/dev/null | grep -iE "^Model name|^CPU\(s\)" || sysctl -n machdep.cpu.brand_string 2>/dev/null || wmic cpu get name 2>nul' || true
		rsh "ls $MDIR 2>/dev/null | grep -c gguf | sed 's/^/models present: /'" 2>/dev/null || true
	else
		echo "  UNREACHABLE"
	fi
	echo
}

cmd="${1:-}"; shift || true
case "$cmd" in
	check)   t="${1:-all}"; if [ "$t" = all ]; then for h in $(all_hosts); do check_host "$h"; done; else check_host "$t"; fi ;;
	deploy)  t="${1:?host}"; if [ "$t" = all ]; then for h in $(all_hosts); do deploy "$h"; done; else deploy "$t"; fi ;;
	run)     run_host "$@" ;;
	bench)   t="${1:-all}"; if [ "$t" = all ]; then for h in $(all_hosts); do bench_host "$h"; done; else bench_host "$t"; fi ;;
	profile) profile_host "$@" ;;
	models)  resolve_models | tr ' ' '\n' | sed '/^$/d' ;;
	*) sed -n '2,40p' "$0"; exit 1 ;;
esac
