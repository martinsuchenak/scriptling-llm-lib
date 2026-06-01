#!/usr/bin/env bash
#
# fleet.sh — cross-compile the `infer` binary, push it to remote hosts (models
# stay on the remote), run benchmarks / CPU profiles, and collect results.
#
# The remote needs NOTHING but the pushed binary and the model files — no Go
# toolchain. Cross-compilation happens here; GOAMD64 is left at v1 so a single
# linux/amd64 build runs on every x86 box (incl. the no-AVX Xeon X5675) — the
# library selects SIMD kernels at runtime via CPUID.
#
# Supports linux, darwin, and windows targets. Windows hosts are driven over
# OpenSSH assuming the default cmd.exe shell; write their paths in the config
# with forward slashes (e.g. C:/models).
#
# Usage:
#   ./bench/fleet.sh check [host|all]              connectivity + CPU info
#   ./bench/fleet.sh deploy <host|all>             build + copy the binary
#   ./bench/fleet.sh run <host> <model> [args...]  one generation, raw output
#   ./bench/fleet.sh bench [host|all]              standard t/s matrix table
#   ./bench/fleet.sh profile <host> <model> [toks] CPU profile -> pprof -top
#
# <model> is a .gguf basename resolved against the host's model_dir (or an
# absolute path). Env overrides: FLEET_CONF, FLEET_PROMPT, FLEET_TOKENS,
# SSH_OPTS, MODELS (space-separated basenames for `bench`).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONF="${FLEET_CONF:-$HERE/hosts.conf}"
BINDIR="$HERE/bin"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8}"
PROMPT="${FLEET_PROMPT:-Write a long, detailed story about a dog and its adventures across the world.}"
TOKENS="${FLEET_TOKENS:-120}"
MODELS="${MODELS:-SmolLM2-135M-Instruct-Q8_0.gguf SmolLM2-360M-Instruct-Q8_0.gguf SmolLM2-1.7B-Instruct-Q8_0.gguf SmolLM2-1.7B-Instruct-Q4_0.gguf}"

die() { echo "fleet: $*" >&2; exit 1; }

# Populate NAME/SSH/GOOS/GOARCH/RDIR/MDIR for a host name.
load_host() {
	local want="$1" line
	line="$(awk -v n="$want" '!/^[[:space:]]*#/ && NF>=6 && $1==n {print; exit}' "$CONF")"
	[ -n "$line" ] || die "host '$want' not found in $CONF"
	# shellcheck disable=SC2086
	set -- $line
	NAME="$1"; SSH="$2"; GOOS="$3"; GOARCH="$4"; RDIR="$5"; MDIR="$6"
}

all_hosts() { awk '!/^[[:space:]]*#/ && NF>=6 {print $1}' "$CONF"; }
is_local()  { [ "$SSH" = "local" ]; }
is_win()    { [ "$GOOS" = "windows" ]; }
bin_name()  { is_win && echo "infer.exe" || echo "infer"; }

# Build (cached) for a GOOS/GOARCH; echo the local binary path.
build_for() {
	local out="$BINDIR/${1}_${2}/$(GOOS=$1 bin_name)"
	mkdir -p "$(dirname "$out")"
	( cd "$ROOT" && GOOS="$1" GOARCH="$2" go build -o "$out" ./examples/infer )
	echo "$out"
}

remote_sh() { if is_local; then bash -c "$1"; else ssh $SSH_OPTS "$SSH" "$1"; fi; }

# Quote one argument for the remote shell (cmd.exe vs POSIX sh).
q() { if is_win; then printf '"%s"' "$1"; else printf "'%s'" "$1"; fi; }
# Path with the remote separator (cmd needs backslashes to launch the exe).
winpath() { printf '%s' "$1" | tr '/' '\\'; }

resolve_model() { case "$1" in /*|[A-Za-z]:*) echo "$1";; *) echo "$MDIR/$1";; esac; }
remote_bin() { if is_local; then build_for "$GOOS" "$GOARCH"; else echo "$RDIR/$(bin_name)"; fi; }

deploy() {
	load_host "$1"
	local bin; bin="$(build_for "$GOOS" "$GOARCH")"
	if is_local; then echo "[$NAME] local build: $bin"; return; fi
	if is_win; then
		ssh $SSH_OPTS "$SSH" "if not exist \"$(winpath "$RDIR")\" mkdir \"$(winpath "$RDIR")\"" || die "[$NAME] ssh failed"
	else
		ssh $SSH_OPTS "$SSH" "mkdir -p '$RDIR'" || die "[$NAME] ssh failed"
	fi
	scp $SSH_OPTS -q "$bin" "$SSH:$RDIR/$(bin_name)" || die "[$NAME] scp failed"
	is_win || ssh $SSH_OPTS "$SSH" "chmod +x '$RDIR/$(bin_name)'"
	echo "[$NAME] deployed -> $SSH:$RDIR/$(bin_name)"
}

# Build the remote invocation string with the right quoting/paths.
infer_cmd() { # $1=model $2=tokens $3=prof(optional) ...rest extra args
	local model="$1" toks="$2" prof="${3:-}"; shift 3 || true
	local bin; bin="$(remote_bin)"
	if is_win; then
		local c; c="$(q "$(winpath "$bin")") -model $(q "$model") -prompt $(q "$PROMPT") -tokens $toks -strategy greedy"
		[ -n "$prof" ] && c="$c -prof $(q "$(winpath "$prof")")"
		printf '%s %s' "$c" "$*"
	else
		local c; c="$(q "$bin") -model $(q "$model") -prompt $(q "$PROMPT") -tokens $toks -strategy greedy"
		[ -n "$prof" ] && c="$c -prof $(q "$prof")"
		printf '%s %s' "$c" "$*"
	fi
}

run_host() {
	load_host "$1"; shift
	local model; model="$(resolve_model "$1")"; shift
	remote_sh "$(infer_cmd "$model" "$TOKENS" "" "$@")"
}

bench_host() {
	load_host "$1"; deploy "$1" >/dev/null
	printf '== %-7s (%s/%s, %s) ==\n' "$NAME" "$GOOS" "$GOARCH" "$SSH"
	printf '%-38s %10s %10s\n' "model" "prefill" "decode"
	local m out pre dec
	for m in $MODELS; do
		if out="$(remote_sh "$(infer_cmd "$(resolve_model "$m")" "$TOKENS" "")" 2>&1)"; then
			pre="$(printf '%s\n' "$out" | awk '/prefill/{print $(NF-1)" "$NF}')"
			dec="$(printf '%s\n' "$out" | awk '/decode/{print $(NF-1)" "$NF}')"
			printf '%-38s %10s %10s\n' "$m" "${pre:-?}" "${dec:-?}"
		else
			printf '%-38s %10s\n' "$m" "ERR (missing model?)"
		fi
	done
	echo
}

profile_host() {
	load_host "$1"; deploy "$1" >/dev/null
	local model toks rprof lprof
	model="$(resolve_model "$2")"; toks="${3:-150}"
	rprof="$RDIR/cpu.prof"; is_local && rprof="/tmp/fleet-$NAME.prof"
	echo "[$NAME] profiling $2 ($toks tokens)..."
	remote_sh "$(infer_cmd "$model" "$toks" "$rprof")" >/dev/null 2>&1 || true
	lprof="$BINDIR/${NAME}.prof"
	if is_local; then cp "$rprof" "$lprof"; else scp $SSH_OPTS -q "$SSH:$rprof" "$lprof" || die "[$NAME] could not fetch profile"; fi
	echo "[$NAME] top functions:"
	go tool pprof -top -nodecount=20 "$(build_for "$GOOS" "$GOARCH")" "$lprof" 2>/dev/null | tail -24
}

check_host() {
	load_host "$1"
	printf '== %-7s (%s, %s/%s) ==\n' "$NAME" "$SSH" "$GOOS" "$GOARCH"
	if remote_sh 'uname -sm 2>/dev/null || ver' >/tmp/.fleetchk 2>/dev/null; then
		cat /tmp/.fleetchk
		remote_sh 'lscpu 2>/dev/null | grep -iE "^Model name|^CPU\(s\)" || sysctl -n machdep.cpu.brand_string 2>/dev/null || wmic cpu get name 2>nul' || true
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
	*) sed -n '2,34p' "$0"; exit 1 ;;
esac
