package scriptlingllmlib

import (
	"testing"
)

func TestChatTemplateJinja2Detection(t *testing.T) {
	jinjaTpl := `{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}<|im_start|>assistant
`
	result := applyChatTemplate(jinjaTpl, "hello", "")
	if result != "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n" {
		t.Errorf("jinja2 template result = %q", result)
	}
}

func TestChatTemplateJinja2WithSystem(t *testing.T) {
	tpl := "{{ messages }}"
	result := applyChatTemplate(tpl, "test prompt", "custom system")
	if result != "<|im_start|>system\ncustom system<|im_end|>\n<|im_start|>user\ntest prompt<|im_end|>\n<|im_start|>assistant\n" {
		t.Errorf("jinja2 with system = %q", result)
	}
}

func TestChatTemplateSimpleSubstitution(t *testing.T) {
	tpl := "User: {prompt}\nAssistant:"
	result := applyChatTemplate(tpl, "hello", "")
	if result != "User: hello\nAssistant:" {
		t.Errorf("simple template = %q", result)
	}
}

func TestChatTemplateSystemPrompt(t *testing.T) {
	tpl := "System: {system_prompt}\nUser: {prompt}\nAssistant:"
	result := applyChatTemplate(tpl, "hi", "be helpful")
	if result != "System: be helpful\nUser: hi\nAssistant:" {
		t.Errorf("system template = %q", result)
	}
}

func TestChatTemplateConditionalWithSystem(t *testing.T) {
	tpl := "{% if system_prompt %}System: {system_prompt}\n{% endif %}User: {prompt}\nAssistant:"
	result := applyChatTemplate(tpl, "hi", "be helpful")
	if result != "System: be helpful\nUser: hi\nAssistant:" {
		t.Errorf("conditional with system = %q", result)
	}
}

func TestChatTemplateConditionalNoSystem(t *testing.T) {
	tpl := "{% if system_prompt %}System: {system_prompt}\n{% endif %}User: {prompt}\nAssistant:"
	result := applyChatTemplate(tpl, "hi", "")
	if result != "User: hi\nAssistant:" {
		t.Errorf("conditional without system = %q", result)
	}
}

func TestChatTemplateDefaultTemplates(t *testing.T) {
	smollm := defaultTemplates["smollm2"]
	if smollm == "" {
		t.Error("smollm2 template missing")
	}
	chatml := defaultTemplates["chatml"]
	if chatml == "" {
		t.Error("chatml template missing")
	}
}
