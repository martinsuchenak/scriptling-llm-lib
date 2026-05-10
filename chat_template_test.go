package scriptlingllmlib

import (
	"strings"
	"testing"
)

func TestChatTemplateLlama2(t *testing.T) {
	tpl := `{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' }}{% if loop.first %}<<SYS>>
{{ system_prompt }}
<</SYS>>

{% endif %}{{ message['content'] + ' [/INST]' }}{% endif %}{% if message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' }}{% endif %}{% endfor %}`
	result := applyChatTemplate(tpl, "hello", "", "")
	if !strings.Contains(result, "[INST]") {
		t.Errorf("llama2 template should contain [INST], got %q", result)
	}
	if !strings.Contains(result, "<<SYS>>") {
		t.Errorf("llama2 template should contain <<SYS>>, got %q", result)
	}
	if !strings.Contains(result, "hello") {
		t.Errorf("llama2 template should contain prompt, got %q", result)
	}
	if !strings.Contains(result, "[/INST]") {
		t.Errorf("llama2 template should contain [/INST], got %q", result)
	}
}

func TestChatTemplateLlama3(t *testing.T) {
	tpl := `{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}`
	result := applyChatTemplate(tpl, "hello", "be helpful", "")
	if !strings.Contains(result, "<|start_header_id|>user<|end_header_id|>") {
		t.Errorf("llama3 template should contain user header, got %q", result)
	}
	if !strings.Contains(result, "hello") {
		t.Errorf("llama3 template should contain prompt, got %q", result)
	}
	if !strings.Contains(result, "<|start_header_id|>assistant<|end_header_id|>") {
		t.Errorf("llama3 template should contain assistant header, got %q", result)
	}
	if !strings.Contains(result, "<|eot_id|>") {
		t.Errorf("llama3 template should contain eot_id, got %q", result)
	}
}

func TestChatTemplateChatMLJinja2(t *testing.T) {
	tpl := `{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if message['role'] != 'assistant' %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`
	result := applyChatTemplate(tpl, "hello", "", "")
	if !strings.Contains(result, "<|im_start|>user") {
		t.Errorf("chatml jinja should contain user start, got %q", result)
	}
	if !strings.Contains(result, "hello") {
		t.Errorf("chatml jinja should contain prompt, got %q", result)
	}
	if !strings.Contains(result, "<|im_start|>assistant") {
		t.Errorf("chatml jinja should contain assistant start, got %q", result)
	}
}

func TestChatTemplateJinja2Qwen3(t *testing.T) {
	tpl := `{% for message in messages %}{{ message.content }}{% endfor %}`
	result := applyChatTemplate(tpl, "hello", "", "qwen3")
	if !strings.Contains(result, "/no_think") {
		t.Errorf("qwen3 template should contain /no_think, got %q", result)
	}
	result2 := applyChatTemplate(tpl, "hello", "", "llama")
	if strings.Contains(result2, "/no_think") {
		t.Errorf("llama template should NOT contain /no_think, got %q", result2)
	}
}

func TestChatTemplateJinja2WithSystem(t *testing.T) {
	tpl := `{% for message in messages %}{{ message['content'] }}{% endfor %}`
	result := applyChatTemplate(tpl, "test prompt", "custom system", "")
	if !strings.Contains(result, "test prompt") {
		t.Errorf("jinja2 with system = %q", result)
	}
}

func TestChatTemplateSimpleSubstitution(t *testing.T) {
	tpl := "User: {prompt}\nAssistant:"
	result := applyChatTemplate(tpl, "hello", "", "")
	if result != "User: hello\nAssistant:" {
		t.Errorf("simple template = %q", result)
	}
}

func TestChatTemplateSystemPrompt(t *testing.T) {
	tpl := "System: {system_prompt}\nUser: {prompt}\nAssistant:"
	result := applyChatTemplate(tpl, "hi", "be helpful", "")
	if result != "System: be helpful\nUser: hi\nAssistant:" {
		t.Errorf("system template = %q", result)
	}
}

func TestChatTemplateConditionalWithSystem(t *testing.T) {
	tpl := "{% if system_prompt %}System: {system_prompt}\n{% endif %}User: {prompt}\nAssistant:"
	result := applyChatTemplate(tpl, "hi", "be helpful", "")
	if result != "System: be helpful\nUser: hi\nAssistant:" {
		t.Errorf("conditional with system = %q", result)
	}
}

func TestChatTemplateConditionalNoSystem(t *testing.T) {
	tpl := "{% if system_prompt %}System: {system_prompt}\n{% endif %}User: {prompt}\nAssistant:"
	result := applyChatTemplate(tpl, "hi", "", "")
	if result != "User: hi\nAssistant:" {
		t.Errorf("conditional without system = %q", result)
	}
}

func TestChatTemplateDefaultTemplates(t *testing.T) {
	for _, name := range []string{"smollm2", "chatml", "llama2", "llama3", "llama", "mistral"} {
		if _, ok := defaultTemplates[name]; !ok {
			t.Errorf("missing template: %s", name)
		}
	}
}

func TestContinuationLlama2(t *testing.T) {
	tpl := `{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' }}{% if loop.first %}<<SYS>>{{ system_prompt }}<</SYS>>{% endif %}{{ message['content'] + ' [/INST]' }}{% endif %}{% endfor %}`
	result := applyContinuation(tpl, "follow up", "")
	if !strings.Contains(result, "[INST]") {
		t.Errorf("llama2 continuation should contain [INST], got %q", result)
	}
	if !strings.Contains(result, "follow up") {
		t.Errorf("llama2 continuation should contain prompt, got %q", result)
	}
	if strings.Contains(result, "<<SYS>>") {
		t.Errorf("llama2 continuation should NOT contain system, got %q", result)
	}
}

func TestContinuationLlama3(t *testing.T) {
	tpl := `{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' }}`
	result := applyContinuation(tpl, "follow up", "")
	if !strings.Contains(result, "<|start_header_id|>user") {
		t.Errorf("llama3 continuation should contain user header, got %q", result)
	}
	if !strings.Contains(result, "follow up") {
		t.Errorf("llama3 continuation should contain prompt, got %q", result)
	}
}

func TestContinuationChatML(t *testing.T) {
	tpl := `<|im_start|>system`
	result := applyContinuation(tpl, "follow up", "")
	if !strings.Contains(result, "<|im_start|>user") {
		t.Errorf("chatml continuation should contain user start, got %q", result)
	}
	if !strings.Contains(result, "follow up") {
		t.Errorf("chatml continuation should contain prompt, got %q", result)
	}
}

func TestJinjaLoopFirst(t *testing.T) {
	tpl := `{% for message in messages %}{% if loop.first %}FIRST:{% endif %}{{ message['content'] }}{% endfor %}`
	result := renderJinja(tpl, "", "hello")
	if !strings.Contains(result, "FIRST:") {
		t.Errorf("loop.first should be true on first iteration, got %q", result)
	}
	if !strings.Contains(result, "hello") {
		t.Errorf("should contain prompt, got %q", result)
	}
}

func TestJinjaSetSlice(t *testing.T) {
	tpl := `{% set loop_messages = messages[1:] %}{% for message in loop_messages %}{{ message['content'] }}{% endfor %}`
	ctx := newJinjaCtx("sys", "hello")
	ctx.messages = []map[string]string{
		{"role": "system", "content": "sys"},
		{"role": "user", "content": "hello"},
	}
	result := renderJinjaWithCtx(tpl, ctx)
	if !strings.Contains(result, "hello") {
		t.Errorf("should contain user message, got %q", result)
	}
	if strings.Contains(result, "sys") {
		t.Errorf("should NOT contain system message, got %q", result)
	}
}

func TestJinjaConcatFilter(t *testing.T) {
	tpl := `{% for message in messages %}{{ message['content'] | trim + '!' }}{% endfor %}`
	result := renderJinja(tpl, "", "hello")
	if !strings.Contains(result, "hello!") {
		t.Errorf("should contain 'hello!', got %q", result)
	}
}

func TestJinjaConditionalRole(t *testing.T) {
	tpl := `{% for message in messages %}{% if message['role'] == 'user' %}USER:{{ message['content'] }}{% elif message['role'] == 'assistant' %}ASST:{{ message['content'] }}{% endif %}{% endfor %}`
	result := renderJinja(tpl, "", "hello")
	if result != "USER:hello" {
		t.Errorf("role conditional = %q", result)
	}
}

func TestJinjaElseBranch(t *testing.T) {
	tpl := `{% for message in messages %}{% if message['role'] == 'system' %}SYS{% else %}OTHER:{{ message['content'] }}{% endif %}{% endfor %}`
	result := renderJinja(tpl, "", "hello")
	if result != "OTHER:hello" {
		t.Errorf("else branch = %q", result)
	}
}
