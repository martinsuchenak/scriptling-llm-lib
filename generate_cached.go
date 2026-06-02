package scriptlingllmlib

import "context"

// GenerateWithCache runs a generation against a cached model. It is safe to call
// concurrently: each call runs on its own clone of the shared (read-only) model
// weights, and turns of the same sessionID are serialized.
func GenerateWithCache(
	modelPath string,
	prompt string,
	maxTokens int,
	strategy string,
	temperature float64,
	topK int,
	topP float64,
	repeatPenalty float64,
	repeatLastN int,
	systemPrompt string,
	templateName string,
	sessionID string,
) (string, int, int, float64, float64, error) {
	return GenerateWithCacheContext(
		context.Background(), modelPath, prompt, maxTokens, strategy, temperature,
		topK, topP, repeatPenalty, repeatLastN, systemPrompt, templateName, sessionID,
	)
}

// GenerateWithCacheContext is GenerateWithCache with cancellation. When ctx is
// cancelled (client disconnect, deadline) the decode loop stops between tokens
// and returns the partial text generated so far together with ctx.Err(). Prefill
// is not interruptible, so cancellation takes effect once decoding begins.
func GenerateWithCacheContext(
	ctx context.Context,
	modelPath string,
	prompt string,
	maxTokens int,
	strategy string,
	temperature float64,
	topK int,
	topP float64,
	repeatPenalty float64,
	repeatLastN int,
	systemPrompt string,
	templateName string,
	sessionID string,
) (string, int, int, float64, float64, error) {
	return GenerateWithCacheStream(
		ctx, modelPath, prompt, maxTokens, strategy, temperature,
		topK, topP, repeatPenalty, repeatLastN, systemPrompt, templateName, sessionID, nil,
	)
}

// GenerateWithCacheStream is GenerateWithCacheContext with streaming. If onToken
// is non-nil it is called with each decoded text delta as tokens are produced
// (byte-level tokens are buffered so every delta is valid UTF-8). The full text
// is still returned at the end. onToken runs on the calling goroutine inside the
// decode loop, so keep it fast; it must not call back into this model. Pass a nil
// onToken for non-streaming behaviour.
func GenerateWithCacheStream(
	ctx context.Context,
	modelPath string,
	prompt string,
	maxTokens int,
	strategy string,
	temperature float64,
	topK int,
	topP float64,
	repeatPenalty float64,
	repeatLastN int,
	systemPrompt string,
	templateName string,
	sessionID string,
	onToken func(string),
) (string, int, int, float64, float64, error) {
	shared, err := globalModelCacheF32.getOrLoad(modelPath)
	if err != nil {
		return "", 0, 0, 0, 0, err
	}

	result, nGen, nPrompt, prefillMs, decodeMs := runGenerate(
		ctx, onToken, shared, modelPath, prompt, maxTokens, strategy, temperature,
		topK, topP, repeatPenalty, repeatLastN, systemPrompt, templateName, sessionID,
	)
	return result, nGen, nPrompt, prefillMs, decodeMs, ctx.Err()
}

func ClearSessionWithCache(modelPath string, sessionID string) {
	globalModelCacheF32.clearSession(modelPath, sessionID)
}
