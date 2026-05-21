package scriptlingllmlib

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
	model, err := globalModelCacheF32.getOrLoad(modelPath)
	if err != nil {
		return "", 0, 0, 0, 0, err
	}

	kvStartPos := 0
	if sessionID != "" {
		globalModelCacheF32.mu.Lock()
		entry := globalModelCacheF32.getSession(modelPath, sessionID)
		if entry != nil {
			model.KVCaches = entry.kvCaches
			kvStartPos = entry.kvPos
		} else {
			model.initKVCaches()
		}
		globalModelCacheF32.mu.Unlock()
	}

	result, nGen, nPrompt, finalPos := model.Generate(
		prompt, maxTokens, strategy, temperature,
		topK, topP, repeatPenalty, repeatLastN,
		systemPrompt, templateName, kvStartPos,
	)

	if sessionID != "" {
		globalModelCacheF32.mu.Lock()
		globalModelCacheF32.saveSession(modelPath, sessionID, model.KVCaches, finalPos)
		globalModelCacheF32.mu.Unlock()
	}

	return result, nGen, nPrompt, model.PrefillMs, model.DecodeMs, nil
}

func ClearSessionWithCache(modelPath string, sessionID string) {
	globalModelCacheF32.mu.Lock()
	globalModelCacheF32.clearSession(modelPath, sessionID)
	globalModelCacheF32.mu.Unlock()
}


