package scriptlingllmlib

import "context"

// Sampling strategies accepted by GenerateOptions.Strategy.
const (
	StrategyGreedy      = "greedy"
	StrategyTemperature = "temperature"
	StrategyTopK        = "top_k"
	StrategyTopP        = "top_p"
)

// GenerateOptions configures a single generation. Only Model and Prompt are
// required; zero-valued fields fall back to the defaults noted below. It is the
// ergonomic alternative to the positional GenerateWithCache* functions.
type GenerateOptions struct {
	Model  string // path to the .gguf model (required)
	Prompt string // user prompt (required)

	MaxTokens   int     // max tokens to generate (default 256)
	Strategy    string  // sampling strategy; one of the Strategy* constants (default StrategyGreedy)
	Temperature float64 // sampling temperature, non-greedy only (default 1.0)
	TopK        int     // top-k cutoff for StrategyTopK (default 40)
	TopP        float64 // nucleus cutoff for StrategyTopP (default 0.95)

	// RepeatPenalty divides the logits of recently seen tokens. 1.0 disables it;
	// default 1.1. RepeatLastN is how many recent tokens it applies to (default 64).
	RepeatPenalty float64
	RepeatLastN   int

	System   string // system prompt (optional)
	Template string // chat template name override (optional; e.g. "chatml")

	// Session, when non-empty, persists the KV cache between calls under this id
	// so a multi-turn conversation does not reprocess prior context. Turns of one
	// session are serialized; different sessions run concurrently.
	Session string

	// Context cancels generation between decode tokens; nil means no cancellation.
	Context context.Context

	// OnToken, if set, receives each decoded text delta as it is produced (each
	// is valid UTF-8). The full text is still returned in GenerateResult.Text.
	OnToken func(delta string)
}

// GenerateResult is the outcome of a Generate call.
type GenerateResult struct {
	Text            string  // generated text (no prompt)
	GeneratedTokens int     // number of tokens produced
	PromptTokens    int     // number of tokens in the (templated) prompt
	PrefillMs       float64 // time spent processing the prompt, ms
	DecodeMs        float64 // time spent generating tokens, ms
}

func (o GenerateOptions) withDefaults() GenerateOptions {
	if o.Context == nil {
		o.Context = context.Background()
	}
	if o.Strategy == "" {
		o.Strategy = StrategyGreedy
	}
	if o.MaxTokens <= 0 {
		o.MaxTokens = 256
	}
	if o.Temperature <= 0 {
		o.Temperature = 1.0
	}
	if o.TopK <= 0 {
		o.TopK = 40
	}
	if o.TopP <= 0 {
		o.TopP = 0.95
	}
	if o.RepeatPenalty <= 0 {
		o.RepeatPenalty = 1.1
	}
	if o.RepeatLastN <= 0 {
		o.RepeatLastN = 64
	}
	return o
}

// Generate runs a generation against the global model cache using an options
// struct, returning a structured result. It is the recommended entry point. Like
// the GenerateWithCache* functions it is safe to call concurrently. If
// opts.Context is cancelled, the returned error is ctx.Err() and Text holds the
// partial output generated so far.
func Generate(opts GenerateOptions) (GenerateResult, error) {
	o := opts.withDefaults()
	text, nGen, nPrompt, prefillMs, decodeMs, err := GenerateWithCacheStream(
		o.Context, o.Model, o.Prompt, o.MaxTokens, o.Strategy, o.Temperature,
		o.TopK, o.TopP, o.RepeatPenalty, o.RepeatLastN, o.System, o.Template, o.Session, o.OnToken,
	)
	return GenerateResult{
		Text:            text,
		GeneratedTokens: nGen,
		PromptTokens:    nPrompt,
		PrefillMs:       prefillMs,
		DecodeMs:        decodeMs,
	}, err
}
