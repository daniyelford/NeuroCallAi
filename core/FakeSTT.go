package core

import (
	"context"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewFakeSTT() *FakeSTT {
	return &FakeSTT{}
}
func (f *FakeSTT) Transcribe(
	ctx context.Context,
	segment neurocall.AudioSegment,
) (neurocall.Transcript, error) {
	select {
	case <-ctx.Done():
		return neurocall.Transcript{}, ctx.Err()
	default:
	}

	if len(segment.Data) == 0 {
		return neurocall.Transcript{}, neurocall.ErrInvalidAudio
	}

	return neurocall.Transcript{
		Text:       "hello from fake stt",
		Confidence: 0.99,
		Language:   "en",
	}, nil
}
