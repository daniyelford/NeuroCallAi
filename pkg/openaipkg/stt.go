package openaipkg

import (
	"bytes"
	"context"
	"fmt"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
	openai "github.com/openai/openai-go/v3"
)

type STTProvider struct {
	client *openai.Client
	model  openai.AudioModel
}

func NewSTTProvider(
	client *openai.Client,
) *STTProvider {

	return &STTProvider{
		client: client,
		model:  openai.AudioModelWhisper1,
	}
}

func (s *STTProvider) Transcribe(
	ctx context.Context,
	segment neurocall.AudioSegment,
) (neurocall.Transcript, error) {

	if len(segment.Data) == 0 {
		return neurocall.Transcript{},
			fmt.Errorf("empty audio")
	}

	wav := PCM16ToWAV(
		segment.Data,
		segment.SampleRate,
		segment.Channels,
	)

	resp, err := s.client.Audio.Transcriptions.New(
		ctx,
		openai.AudioTranscriptionNewParams{
			File:  bytes.NewReader(wav),
			Model: s.model,
		},
	)

	if err != nil {
		return neurocall.Transcript{}, err
	}

	return neurocall.Transcript{
		Text:       resp.Text,
		Start:      segment.Start.Milliseconds(),
		End:        segment.End.Milliseconds(),
		Confidence: 1,
	}, nil
}
