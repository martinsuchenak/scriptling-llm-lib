package scriptlingllmlib

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/paularlott/scriptling/errors"
	"github.com/paularlott/scriptling/object"
)

var sampleRng *rand.Rand

func init() {
	sampleRng = rand.New(rand.NewSource(time.Now().UnixNano()))
}

func fnSample(ctx context.Context, kwargs object.Kwargs, args ...object.Object) object.Object {
	if err := errors.RangeArgs(args, 1, 4); err != nil {
		return err
	}

	logits, errObj := toFloatList(args[0], "sample", "logits")
	if errObj != nil {
		return errObj
	}
	n := len(logits)
	if n == 0 {
		return errors.NewError("sample: logits cannot be empty")
	}

	strategy := "greedy"
	if len(args) >= 2 {
		s, ok := args[1].(*object.String)
		if !ok {
			return errors.NewTypeError("STRING", args[1].Type().String())
		}
		strategy = s.Value
	}

	temperature := 1.0
	temperatureObj := kwargs.Get("temperature")
	if temperatureObj != nil {
		t, err := temperatureObj.AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", temperatureObj.Type().String())
		}
		if t <= 0 {
			return errors.NewError("sample: temperature must be positive")
		}
		temperature = t
	} else if len(args) >= 3 {
		t, err := args[2].AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", args[2].Type().String())
		}
		if t <= 0 {
			return errors.NewError("sample: temperature must be positive")
		}
		temperature = t
	}

	topK := 0
	topKObj := kwargs.Get("top_k")
	if topKObj != nil {
		k, err := topKObj.AsInt()
		if err != nil {
			return errors.NewTypeError("INTEGER", topKObj.Type().String())
		}
		topK = int(k)
	} else if len(args) >= 4 {
		k, err := args[3].AsInt()
		if err != nil {
			return errors.NewTypeError("INTEGER", args[3].Type().String())
		}
		topK = int(k)
	}

	topP := 1.0
	topPObj := kwargs.Get("top_p")
	if topPObj != nil {
		p, err := topPObj.AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", topPObj.Type().String())
		}
		if p <= 0 || p > 1.0 {
			return errors.NewError("sample: top_p must be in (0, 1]")
		}
		topP = p
	}

	repeatPenalty := 0.0
	repeatPenaltyObj := kwargs.Get("repeat_penalty")
	if repeatPenaltyObj != nil {
		rp, err := repeatPenaltyObj.AsFloat()
		if err != nil {
			return errors.NewTypeError("INTEGER or FLOAT", repeatPenaltyObj.Type().String())
		}
		if rp < 1.0 {
			return errors.NewError("sample: repeat_penalty must be >= 1.0")
		}
		repeatPenalty = rp
	}

	repeatLastN := 64
	repeatLastNObj := kwargs.Get("repeat_last_n")
	if repeatLastNObj != nil {
		rln, err := repeatLastNObj.AsInt()
		if err != nil {
			return errors.NewTypeError("INTEGER", repeatLastNObj.Type().String())
		}
		if rln > 0 {
			repeatLastN = int(rln)
		}
	}

	var recentTokens []int64
	recentObj := kwargs.Get("recent_tokens")
	if recentObj != nil {
		if list, ok := recentObj.(*object.List); ok {
			recentTokens = make([]int64, len(list.Elements))
			for i, el := range list.Elements {
				v, err := el.AsInt()
				if err != nil {
					return errors.NewTypeError("INTEGER", el.Type().String())
				}
				recentTokens[i] = v
			}
		}
	}

	if repeatPenalty > 0 && len(recentTokens) > 0 {
		applyRepeatPenalty(logits, recentTokens, repeatPenalty, repeatLastN)
	}

	switch strategy {
	case "greedy":
		return sampleGreedy(logits)
	case "temperature":
		return sampleTemperature(logits, temperature, n)
	case "top_k":
		if topK <= 0 {
			topK = 50
		}
		return sampleTopK(logits, temperature, topK, n)
	case "top_p":
		return sampleTopP(logits, temperature, topP, n)
	default:
		return errors.NewError("sample: unknown strategy %q (expected greedy, temperature, top_k, top_p)", strategy)
	}
}

func sampleGreedy(logits []float64) object.Object {
	bestIdx := 0
	bestVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			bestIdx = i
		}
	}
	return &object.Integer{Value: int64(bestIdx)}
}

func sampleTemperature(logits []float64, temperature float64, n int) object.Object {
	probs := softmax(logits, temperature)
	idx := weightedSample(probs, n)
	return &object.Integer{Value: int64(idx)}
}

func sampleTopK(logits []float64, temperature float64, k, n int) object.Object {
	if k > n {
		k = n
	}

	indexed := make([]indexedFloat, n)
	for i, v := range logits {
		indexed[i] = indexedFloat{index: i, value: v}
	}
	partialNthElement(indexed, k)

	topK := make([]float64, k)
	topIndices := make([]int, k)
	for i := 0; i < k; i++ {
		topK[i] = indexed[i].value / temperature
		topIndices[i] = indexed[i].index
	}

	probs := softmaxInPlace(topK)
	offset := sampleRng.Float64()
	cum := 0.0
	for i, p := range probs {
		cum += p
		if offset <= cum {
			return &object.Integer{Value: int64(topIndices[i])}
		}
	}
	return &object.Integer{Value: int64(topIndices[k-1])}
}

func sampleTopP(logits []float64, temperature float64, p float64, n int) object.Object {
	scaled := make([]float64, n)
	for i, v := range logits {
		scaled[i] = v / temperature
	}

	probs := softmax(scaled, 1.0)

	type idxProb struct {
		idx  int
		prob float64
	}
	sorted := make([]idxProb, n)
	for i, pr := range probs {
		sorted[i] = idxProb{i, pr}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].prob > sorted[j].prob
	})

	cum := 0.0
	cutoff := 0
	for i, sp := range sorted {
		cum += sp.prob
		cutoff = i
		if cum >= p {
			break
		}
	}
	cutoff++

	filteredProbs := make([]float64, cutoff)
	filteredIndices := make([]int, cutoff)
	for i := 0; i < cutoff; i++ {
		filteredProbs[i] = sorted[i].prob
		filteredIndices[i] = sorted[i].idx
	}

	var sum float64
	for _, pr := range filteredProbs {
		sum += pr
	}
	invSum := 1.0 / sum

	offset := sampleRng.Float64()
	cum = 0.0
	for i, pr := range filteredProbs {
		cum += pr * invSum
		if offset <= cum {
			return &object.Integer{Value: int64(filteredIndices[i])}
		}
	}
	return &object.Integer{Value: int64(filteredIndices[cutoff-1])}
}

func softmax(logits []float64, temperature float64) []float64 {
	n := len(logits)
	result := make([]float64, n)
	maxVal := logits[0] / temperature
	for i := 1; i < n; i++ {
		v := logits[i] / temperature
		if v > maxVal {
			maxVal = v
		}
	}
	var sumExp float64
	for i, v := range logits {
		e := math.Exp(v/temperature - maxVal)
		result[i] = e
		sumExp += e
	}
	invSum := 1.0 / sumExp
	for i := range result {
		result[i] *= invSum
	}
	return result
}

func softmaxInPlace(logits []float64) []float64 {
	n := len(logits)
	maxVal := logits[0]
	for i := 1; i < n; i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
		}
	}
	var sumExp float64
	for i, v := range logits {
		e := math.Exp(v - maxVal)
		logits[i] = e
		sumExp += e
	}
	invSum := 1.0 / sumExp
	for i := range logits {
		logits[i] *= invSum
	}
	return logits
}

func weightedSample(probs []float64, n int) int {
	offset := sampleRng.Float64()
	cum := 0.0
	for i, p := range probs {
		cum += p
		if offset <= cum {
			return i
		}
	}
	return n - 1
}

func applyRepeatPenalty(logits []float64, recentTokens []int64, penalty float64, lastN int) {
	start := len(recentTokens) - int(lastN)
	if start < 0 {
		start = 0
	}
	window := recentTokens[start:]
	seen := make(map[int64]bool, len(window))
	for _, t := range window {
		idx := int(t)
		if idx < 0 || idx >= len(logits) {
			continue
		}
		if seen[t] {
			continue
		}
		seen[t] = true
		if logits[idx] > 0 {
			logits[idx] = logits[idx] / penalty
		} else {
			logits[idx] = logits[idx] * penalty
		}
	}
}

func partialNthElement(data []indexedFloat, k int) {
	n := len(data)
	if n <= 1 {
		return
	}
	pivot := data[n/2].value
	lo, hi := 0, n-1
	for lo <= hi {
		for data[lo].value > pivot {
			lo++
		}
		for data[hi].value < pivot {
			hi--
		}
		if lo <= hi {
			data[lo], data[hi] = data[hi], data[lo]
			lo++
			hi--
		}
	}
	if k <= hi {
		partialNthElement(data[:hi+1], k)
	} else if k >= lo {
		partialNthElement(data[lo:], k-lo)
	}
}
