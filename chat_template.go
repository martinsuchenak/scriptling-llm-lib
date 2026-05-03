package scriptlingllmlib

import (
	"strings"
)

func applyChatTemplate(template string, prompt string, systemPrompt string) string {
	if strings.Contains(template, "{% for ") || strings.Contains(template, "{{") {
		if systemPrompt == "" {
			systemPrompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
		}
		return "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
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

var defaultTemplates = map[string]string{
	"smollm2": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
	"chatml":  "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}
