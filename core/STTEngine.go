package core

import (
	"context"
	"fmt"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewSTTEngine(
	stt neurocall.STT,
) *STTEngine {

	ctx, cancel := context.WithCancel(
		context.Background(),
	)

	return &STTEngine{
		stt:    stt,
		ctx:    ctx,
		cancel: cancel,
	}
}
func (e *STTEngine) SetSTT(
	stt neurocall.STT,
) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.stt = stt
}
func (e *STTEngine) Transcribe(
	ctx context.Context,
	segment neurocall.AudioSegment,
) (neurocall.Transcript, error) {

	e.mu.RLock()
	stt := e.stt
	engineCtx := e.ctx
	e.mu.RUnlock()

	if stt == nil {
		return neurocall.Transcript{}, fmt.Errorf(
			"STT is not configured",
		)
	}

	if ctx == nil {
		ctx = engineCtx
	}

	return stt.Transcribe(
		ctx,
		segment,
	)
}
func (e *STTEngine) Close() {

	e.mu.Lock()

	cancel := e.cancel
	e.cancel = nil

	e.mu.Unlock()

	if cancel != nil {
		cancel()
	}
}
