package scriptlingllmlib

import (
	"strconv"
	"strings"
)

type jtoken struct {
	kind  string
	value string
	ltrim bool
	rtrim bool
}

type loopCtx struct {
	first bool
	last  bool
	index int
}

type jctx struct {
	messages     []map[string]string
	systemPrompt string
	addGenPrompt bool
	bosToken     string
	vars         map[string]interface{}
}

func newJinjaCtx(systemPrompt, userMessage string) *jctx {
	messages := []map[string]string{
		{"role": "user", "content": userMessage},
	}
	return &jctx{
		messages:     messages,
		systemPrompt: systemPrompt,
		addGenPrompt: true,
		bosToken:     "",
		vars:         make(map[string]interface{}),
	}
}

func tokenizeJinja(tpl string) []jtoken {
	var tokens []jtoken
	i := 0
	for i < len(tpl) {
		if i+2 < len(tpl) && tpl[i] == '{' && tpl[i+1] == '#' {
			end := strings.Index(tpl[i+2:], "#}")
			if end < 0 {
				break
			}
			i = i + 2 + end + 2
		} else if i+1 < len(tpl) && tpl[i] == '{' && (tpl[i+1] == '{' || tpl[i+1] == '%') {
			isExpr := tpl[i+1] == '{'
			closeStr := "}}"
			if !isExpr {
				closeStr = "%}"
			}

			ltrim := false
			start := i + 2
			if start < len(tpl) && tpl[start] == '-' {
				ltrim = true
				start++
			}

			end := strings.Index(tpl[start:], closeStr)
			if end < 0 {
				tokens = append(tokens, jtoken{kind: "text", value: tpl[i:]})
				break
			}
			end += start

			content := tpl[start:end]
			rtrim := false
			if !isExpr && len(content) > 0 && content[len(content)-1] == '-' {
				rtrim = true
				content = content[:len(content)-1]
			}

			kind := "expr"
			if !isExpr {
				kind = "block"
			}

			if ltrim && len(tokens) > 0 && tokens[len(tokens)-1].kind == "text" {
				tokens[len(tokens)-1].value = strings.TrimRight(tokens[len(tokens)-1].value, " \t\n\r")
			}

			tokens = append(tokens, jtoken{
				kind:  kind,
				value: strings.TrimSpace(content),
				ltrim: ltrim,
				rtrim: rtrim,
			})

			i = end + 2

			if rtrim {
				for i < len(tpl) && (tpl[i] == ' ' || tpl[i] == '\t' || tpl[i] == '\n' || tpl[i] == '\r') {
					i++
				}
			}
		} else {
			j := len(tpl)
			for k := i; k+1 < len(tpl); k++ {
				if tpl[k] == '{' && (tpl[k+1] == '{' || tpl[k+1] == '%' || tpl[k+1] == '#') {
					j = k
					break
				}
			}
			tokens = append(tokens, jtoken{kind: "text", value: tpl[i:j]})
			i = j
		}
	}
	return tokens
}

func (c *jctx) execute(tokens []jtoken, start, end int, buf *strings.Builder) {
	i := start
	for i < end {
		tok := tokens[i]
		switch tok.kind {
		case "text":
			buf.WriteString(tok.value)
			i++
		case "expr":
			buf.WriteString(c.evalStr(tok.value))
			i++
		case "block":
			body := tok.value
			if strings.HasPrefix(body, "for ") {
				endIdx := findBlockEnd(tokens, i+1, end, "for", "endfor")
				c.execFor(body, tokens, i+1, endIdx, buf)
				i = endIdx + 1
			} else if strings.HasPrefix(body, "if ") {
				endIdx := findBlockEnd(tokens, i+1, end, "if", "endif")
				c.execIf(body, tokens, i+1, endIdx, buf)
				i = endIdx + 1
			} else if strings.HasPrefix(body, "set ") {
				c.execSet(body)
				i++
			} else {
				i++
			}
		}
	}
}

func findBlockEnd(tokens []jtoken, start, outerEnd int, open, close string) int {
	depth := 1
	for i := start; i < len(tokens); i++ {
		if tokens[i].kind != "block" {
			continue
		}
		body := tokens[i].value
		if strings.HasPrefix(body, open+" ") {
			depth++
		}
		if body == close {
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return outerEnd
}

func (c *jctx) execFor(header string, tokens []jtoken, start, end int, buf *strings.Builder) {
	parts := strings.Fields(header)
	if len(parts) < 4 {
		return
	}
	varName := parts[1]
	iterName := parts[3]

	var items []map[string]string
	switch iterName {
	case "messages":
		items = c.messages
	case "loop_messages":
		if v, ok := c.vars["loop_messages"]; ok {
			if arr, ok := v.([]map[string]string); ok {
				items = arr
			}
		}
	}

	for idx, item := range items {
		c.vars[varName] = item
		c.vars["loop"] = &loopCtx{
			first: idx == 0,
			last:  idx == len(items)-1,
			index: idx + 1,
		}
		c.execute(tokens, start, end, buf)
	}
}

func (c *jctx) execIf(header string, tokens []jtoken, start, endifIdx int, buf *strings.Builder) {
	cond := strings.TrimSpace(strings.TrimPrefix(header, "if "))

	type branch struct {
		cond       string
		start, end int
	}
	var branches []branch

	depth := 0
	curStart := start
	curCond := cond

	for i := start; i < endifIdx; i++ {
		if tokens[i].kind != "block" {
			continue
		}
		body := tokens[i].value
		if strings.HasPrefix(body, "if ") {
			depth++
		}
		if body == "endif" {
			if depth > 0 {
				depth--
				continue
			}
		}
		if depth > 0 {
			continue
		}
		if strings.HasPrefix(body, "elif ") {
			branches = append(branches, branch{cond: curCond, start: curStart, end: i})
			curStart = i + 1
			curCond = strings.TrimSpace(strings.TrimPrefix(body, "elif "))
		} else if body == "else" {
			branches = append(branches, branch{cond: curCond, start: curStart, end: i})
			curStart = i + 1
			curCond = ""
		}
	}
	branches = append(branches, branch{cond: curCond, start: curStart, end: endifIdx})

	for _, b := range branches {
		if b.cond == "" {
			c.execute(tokens, b.start, b.end, buf)
			return
		}
		if c.evalBool(b.cond) {
			c.execute(tokens, b.start, b.end, buf)
			return
		}
	}
}

func (c *jctx) execSet(header string) {
	header = strings.TrimPrefix(header, "set ")
	eqIdx := strings.Index(header, "=")
	if eqIdx < 0 {
		return
	}
	varName := strings.TrimSpace(header[:eqIdx])
	expr := strings.TrimSpace(header[eqIdx+1:])

	if strings.HasSuffix(expr, ":]") {
		bracketIdx := strings.Index(expr, "[")
		if bracketIdx >= 0 {
			arrName := expr[:bracketIdx]
			indexStr := expr[bracketIdx+1 : len(expr)-2]
			if arrName == "messages" {
				startIdx := 0
				for _, ch := range indexStr {
					if ch >= '0' && ch <= '9' {
						startIdx = startIdx*10 + int(ch-'0')
					}
				}
				if startIdx < len(c.messages) {
					c.vars[varName] = c.messages[startIdx:]
				} else {
					c.vars[varName] = []map[string]string{}
				}
			}
		}
		return
	}

	switch expr {
	case "true":
		c.vars[varName] = "true"
	case "false":
		c.vars[varName] = "false"
	case "none":
		c.vars[varName] = ""
	default:
		c.vars[varName] = c.evalStr(expr)
	}
}

func (c *jctx) evalStr(expr string) string {
	expr = strings.TrimSpace(expr)

	if len(expr) >= 2 &&
		((expr[0] == '\'' && expr[len(expr)-1] == '\'') ||
			(expr[0] == '"' && expr[len(expr)-1] == '"')) {
		inner := expr[1 : len(expr)-1]
		if !containsUnquoted(inner, expr[0]) {
			return unescapeJinjaStr(inner)
		}
	}

	parts := splitOnPlus(expr)
	if len(parts) > 1 {
		var buf strings.Builder
		for _, p := range parts {
			buf.WriteString(c.evalStr(strings.TrimSpace(p)))
		}
		return buf.String()
	}

	val := c.evalAtom(expr)
	return val
}

func unescapeJinjaStr(s string) string {
	if !strings.ContainsRune(s, '\\') {
		return s
	}
	var buf strings.Builder
	buf.Grow(len(s))
	i := 0
	for i < len(s) {
		if s[i] == '\\' && i+1 < len(s) {
			switch s[i+1] {
			case 'n':
				buf.WriteByte('\n')
				i += 2
			case 't':
				buf.WriteByte('\t')
				i += 2
			case 'r':
				buf.WriteByte('\r')
				i += 2
			case '\\':
				buf.WriteByte('\\')
				i += 2
			case '\'':
				buf.WriteByte('\'')
				i += 2
			case '"':
				buf.WriteByte('"')
				i += 2
			default:
				buf.WriteByte(s[i])
				i++
			}
		} else {
			buf.WriteByte(s[i])
			i++
		}
	}
	return buf.String()
}

func containsUnquoted(s string, quote byte) bool {
	for i := 1; i < len(s); i++ {
		if s[i] == quote {
			return true
		}
	}
	return false
}

func (c *jctx) evalAtom(expr string) string {
	expr = strings.TrimSpace(expr)

	if len(expr) >= 2 &&
		((expr[0] == '\'' && expr[len(expr)-1] == '\'') ||
			(expr[0] == '"' && expr[len(expr)-1] == '"')) {
		inner := expr[1 : len(expr)-1]
		if !containsUnquoted(inner, expr[0]) {
			return unescapeJinjaStr(inner)
		}
	}

	if pipeIdx := findPipe(expr); pipeIdx >= 0 {
		val := c.evalAtom(strings.TrimSpace(expr[:pipeIdx]))
		filter := strings.TrimSpace(expr[pipeIdx+1:])
		switch filter {
		case "trim":
			return strings.TrimSpace(val)
		}
		return val
	}

	switch expr {
	case "system_prompt":
		return c.systemPrompt
	case "bos_token":
		return c.bosToken
	case "add_generation_prompt":
		if c.addGenPrompt {
			return "true"
		}
		return "false"
	case "true":
		return "true"
	case "false":
		return "false"
	case "none":
		return ""
	}

	return c.evalAccess(expr)
}

func (c *jctx) evalAccess(expr string) string {
	if bracketIdx := strings.IndexByte(expr, '['); bracketIdx >= 0 {
		objExpr := expr[:bracketIdx]
		rest := expr[bracketIdx+1:]

		objName := strings.TrimSpace(objExpr)
		var obj interface{}
		switch objName {
		case "message":
			obj = c.vars["message"]
		case "messages":
			obj = c.messages
		default:
			if v, ok := c.vars[objName]; ok {
				obj = v
			}
		}

		if obj == nil {
			return ""
		}

		var key string
		var remaining string

		if len(rest) > 0 && (rest[0] == '\'' || rest[0] == '"') {
			closeQuote := rest[0:1]
			closeIdx := strings.Index(rest[1:], closeQuote)
			if closeIdx >= 0 {
				key = rest[1 : closeIdx+1]
				afterKey := rest[closeIdx+2:]
				if len(afterKey) > 0 && afterKey[0] == '[' {
					remaining = afterKey
				}
			}
			if m, ok := obj.(map[string]string); ok && remaining == "" {
				return m[key]
			}
		} else {
			closeIdx := strings.IndexByte(rest, ']')
			if closeIdx >= 0 {
				idxStr := rest[:closeIdx]
				idx := 0
				for _, ch := range idxStr {
					if ch >= '0' && ch <= '9' {
						idx = idx*10 + int(ch-'0')
					}
				}
				if arr, ok := obj.([]map[string]string); ok {
					if idx < len(arr) {
						elem := arr[idx]
						afterBracket := rest[closeIdx+1:]
						if len(afterBracket) > 0 && afterBracket[0] == '[' {
							return c.evalAccessChain(elem, afterBracket)
						}
						if len(afterBracket) > 0 && afterBracket[0] == '.' {
							prop := afterBracket[1:]
							return elem[prop]
						}
						return ""
					}
				}
				return ""
			}
		}

		if key != "" {
			if m, ok := obj.(map[string]string); ok {
				return m[key]
			}
		}
	}

	if dotIdx := strings.IndexByte(expr, '.'); dotIdx >= 0 {
		objName := expr[:dotIdx]
		prop := expr[dotIdx+1:]

		if objName == "loop" {
			if lp, ok := c.vars["loop"]; ok {
				if lc, ok := lp.(*loopCtx); ok {
					switch prop {
					case "first":
						if lc.first {
							return "true"
						}
						return "false"
					case "last":
						if lc.last {
							return "true"
						}
						return "false"
					case "index":
						return strconv.Itoa(lc.index)
					case "index0":
						return strconv.Itoa(lc.index - 1)
					}
				}
			}
		}

		if v, ok := c.vars[objName]; ok {
			if m, ok := v.(map[string]string); ok {
				return m[prop]
			}
		}
	}

	if v, ok := c.vars[expr]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}

	return ""
}

func (c *jctx) evalAccessChain(obj map[string]string, rest string) string {
	if len(rest) == 0 || rest[0] != '[' {
		return ""
	}
	rest = rest[1:]
	if len(rest) > 0 && (rest[0] == '\'' || rest[0] == '"') {
		closeQuote := rest[0:1]
		closeIdx := strings.Index(rest[1:], closeQuote)
		if closeIdx >= 0 {
			key := rest[1 : closeIdx+1]
			return obj[key]
		}
	}
	return ""
}

func (c *jctx) evalBool(expr string) bool {
	expr = strings.TrimSpace(expr)

	if strings.HasPrefix(expr, "not ") {
		rest := expr[4:]
		if strings.HasSuffix(rest, " is defined") {
			name := strings.TrimSpace(rest[:len(rest)-10])
			_, ok := c.vars[name]
			return !ok
		}
		return !c.evalBool(rest)
	}

	if strings.HasPrefix(expr, "(") && strings.HasSuffix(expr, ")") {
		inner := expr[1 : len(expr)-1]
		if balancedParens(inner) {
			return c.evalBool(inner)
		}
	}

	if orIdx := findLogicalOp(expr, " or "); orIdx >= 0 {
		return c.evalBool(expr[:orIdx]) || c.evalBool(expr[orIdx+4:])
	}

	if andIdx := findLogicalOp(expr, " and "); andIdx >= 0 {
		return c.evalBool(expr[:andIdx]) && c.evalBool(expr[andIdx+5:])
	}

	if eqIdx := findCmp(expr, "=="); eqIdx >= 0 {
		left := c.evalStr(strings.TrimSpace(expr[:eqIdx]))
		right := c.evalStr(strings.TrimSpace(expr[eqIdx+2:]))
		return left == right
	}

	if neqIdx := findCmp(expr, "!="); neqIdx >= 0 {
		left := c.evalStr(strings.TrimSpace(expr[:neqIdx]))
		right := c.evalStr(strings.TrimSpace(expr[neqIdx+2:]))
		return left != right
	}

	if strings.HasSuffix(expr, " is not none") {
		val := c.evalStr(strings.TrimSpace(expr[:len(expr)-11]))
		return val != "" && val != "none"
	}

	if strings.HasSuffix(expr, " is none") {
		val := c.evalStr(strings.TrimSpace(expr[:len(expr)-8]))
		return val == "" || val == "none"
	}

	if strings.HasSuffix(expr, " is defined") {
		name := strings.TrimSpace(expr[:len(expr)-10])
		_, ok := c.vars[name]
		return ok
	}

	val := c.evalStr(expr)
	return val != "" && val != "false" && val != "0"
}

func findCmp(expr string, op string) int {
	for i := 0; i <= len(expr)-len(op); i++ {
		if expr[i:i+len(op)] == op && !inQuotes(expr, i) {
			return i
		}
	}
	return -1
}

func findLogicalOp(expr string, op string) int {
	depth := 0
	for i := 0; i <= len(expr)-len(op); i++ {
		if expr[i] == '(' {
			depth++
		} else if expr[i] == ')' {
			depth--
		} else if depth == 0 && expr[i:i+len(op)] == op && !inQuotes(expr, i) {
			return i
		}
	}
	return -1
}

func balancedParens(s string) bool {
	depth := 0
	for _, ch := range s {
		switch ch {
		case '(':
			depth++
		case ')':
			if depth == 0 {
				return false
			}
			depth--
		}
	}
	return depth == 0
}

func findPipe(expr string) int {
	for i := 0; i < len(expr); i++ {
		if expr[i] == '|' && !inQuotes(expr, i) {
			return i
		}
	}
	return -1
}

func inQuotes(s string, pos int) bool {
	sq := 0
	dq := 0
	for i := 0; i < pos && i < len(s); i++ {
		switch s[i] {
		case '\'':
			if dq%2 == 0 {
				sq++
			}
		case '"':
			if sq%2 == 0 {
				dq++
			}
		}
	}
	return sq%2 == 1 || dq%2 == 1
}

func splitOnPlus(expr string) []string {
	var parts []string
	sq := 0
	dq := 0
	depth := 0
	last := 0

	for i := 0; i < len(expr); i++ {
		ch := expr[i]
		switch {
		case ch == '\'' && dq%2 == 0:
			sq++
		case ch == '"' && sq%2 == 0:
			dq++
		case sq%2 == 0 && dq%2 == 0:
			if ch == '[' {
				depth++
			} else if ch == ']' {
				depth--
			} else if ch == '+' && depth == 0 {
				parts = append(parts, expr[last:i])
				last = i + 1
			}
		}
	}
	parts = append(parts, expr[last:])
	return parts
}

func renderJinja(tpl string, systemPrompt, userMessage string) string {
	tokens := tokenizeJinja(tpl)
	ctx := newJinjaCtx(systemPrompt, userMessage)

	usesSystemVar := strings.Contains(tpl, "{{ system_prompt }}") ||
		strings.Contains(tpl, "{{ system_message }}") ||
		strings.Contains(tpl, "{{system_prompt}}") ||
		strings.Contains(tpl, "{{system_message}}")

	if systemPrompt != "" && !usesSystemVar {
		ctx.messages = append([]map[string]string{
			{"role": "system", "content": systemPrompt},
		}, ctx.messages...)
	}

	var buf strings.Builder
	ctx.execute(tokens, 0, len(tokens), &buf)
	return buf.String()
}

func renderJinjaWithCtx(tpl string, ctx *jctx) string {
	tokens := tokenizeJinja(tpl)
	var buf strings.Builder
	ctx.execute(tokens, 0, len(tokens), &buf)
	return buf.String()
}

func applyChatTemplate(template string, prompt string, systemPrompt string, arch string) string {
	if strings.Contains(template, "{% for ") || strings.Contains(template, "{{") {
		if systemPrompt == "" && !strings.Contains(template, "if system_prompt") && !strings.Contains(template, "if system_message") {
			if arch == "qwen3" {
				systemPrompt = "/no_think"
			} else {
				systemPrompt = "You are a helpful AI assistant."
			}
		}
		return renderJinja(template, systemPrompt, prompt)
	}

	result := template

	result = strings.ReplaceAll(result, "{prompt}", prompt)
	result = strings.ReplaceAll(result, "{system_prompt}", systemPrompt)
	result = strings.ReplaceAll(result, "{system_message}", systemPrompt)

	for {
		startIdx := strings.Index(result, "{% if ")
		if startIdx == -1 {
			break
		}

		endIf := strings.Index(result[startIdx:], "{% endif %}")
		if endIf == -1 {
			break
		}
		endIf += startIdx

		thenMarker := strings.Index(result[startIdx:], "%}")
		if thenMarker == -1 {
			break
		}
		thenMarker += startIdx

		condition := strings.TrimSpace(result[startIdx+6 : thenMarker])
		blockContent := result[thenMarker+2 : endIf]

		if condition == "system_prompt" || condition == "system_message" {
			if systemPrompt != "" {
				result = result[:startIdx] + blockContent + result[endIf+len("{% endif %}"):]
			} else {
				result = result[:startIdx] + result[endIf+len("{% endif %}"):]
			}
		} else {
			result = result[:startIdx] + result[endIf+len("{% endif %}"):]
		}
	}

	return result
}

func applyContinuation(template string, prompt string, arch string) string {
	if strings.Contains(template, "<<SYS>>") {
		return "[INST] " + prompt + " [/INST]"
	}
	if strings.Contains(template, "<|start_header_id|>") {
		return "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	}
	if strings.Contains(template, "<|im_start|>") {
		return "<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
	}
	if strings.Contains(template, "[INST]") {
		return "[INST] " + prompt + " [/INST]"
	}
	return "<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
}

var defaultTemplates = map[string]string{
	"smollm2": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
	"chatml":  "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
	"llama2":  "{% if system_prompt %}[INST] <<SYS>>\n{{ system_prompt }}\n<</SYS>>\n\n{% endif %}{{ prompt }} [/INST]",
	"llama3":  "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
	"llama":   "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
	"mistral": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + '</s>' }}{% endif %}{% endfor %}",
}
