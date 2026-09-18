package core

import (
	"context"
	"fmt"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewTTSPlayback(
	engine *TTSEngine,
) (*TTSPlayback, error) {
	if engine == nil {
		return nil, fmt.Errorf("TTS engine is nil")
	}

	ctx, cancel := context.WithCancel(
		context.Background(),
	)

	return &TTSPlayback{
		engine: engine,
		ctx:    ctx,
		cancel: cancel,
	}, nil
}

func (p *TTSPlayback) Synthesize(
	ctx context.Context,
	text string,
) (neurocall.AudioStreamData, error) {
	if p == nil || p.engine == nil {
		return neurocall.AudioStreamData{}, fmt.Errorf(
			"TTS playback is not configured",
		)
	}

	if ctx == nil {
		ctx = p.ctx
	}

	return p.engine.Synthesize(
		ctx,
		text,
	)
}

func (p *TTSPlayback) Close() error {
	if p == nil {
		return nil
	}

	if p.cancel != nil {
		p.cancel()
	}

	return nil
}
