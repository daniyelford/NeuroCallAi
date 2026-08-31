package core

import (
	"context"
	"fmt"
	"strings"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewTTSEngine(tts neurocall.TTS) *TTSEngine {
	return &TTSEngine{
		tts: tts,
	}
}
func (e *TTSEngine) SetTTS(tts neurocall.TTS) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.tts = tts
}
func (e *TTSEngine) Synthesize(
	ctx context.Context,
	text string,
) (neurocall.AudioStreamData, error) {

	if strings.TrimSpace(text) == "" {
		return neurocall.AudioStreamData{}, fmt.Errorf("text is empty")
	}

	e.mu.RLock()
	tts := e.tts
	e.mu.RUnlock()

	if tts == nil {
		return neurocall.AudioStreamData{}, fmt.Errorf(
			"TTS is not configured",
		)
	}

	return tts.Synthesize(ctx, text)
}
