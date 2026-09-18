package openaipkg

import (
	"context"
	"fmt"
	"io"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
	openai "github.com/openai/openai-go/v3"
)

type OpenAITTS struct {
	client *openai.Client
}

func NewTTS(
	client *openai.Client,
) *OpenAITTS {

	return &OpenAITTS{
		client: client,
	}
}

func (t *OpenAITTS) Synthesize(
	ctx context.Context,
	text string,
) (neurocall.AudioStreamData, error) {

	if text == "" {
		return neurocall.AudioStreamData{},
			fmt.Errorf("empty text")
	}

	resp, err := t.client.Audio.Speech.New(
		ctx,
		openai.AudioSpeechNewParams{

			Model: openai.SpeechModelTTS1,

			Input: text,

			Voice: openai.AudioSpeechNewParamsVoiceUnion{
				OfString: openai.String("alloy"),
			},

			ResponseFormat: openai.AudioSpeechNewParamsResponseFormatPCM,
		},
	)

	if err != nil {
		return neurocall.AudioStreamData{}, err
	}

	data, err := io.ReadAll(resp.Body)

	if err != nil {
		return neurocall.AudioStreamData{}, err
	}

	return neurocall.AudioStreamData{

		Format: neurocall.AudioFormat{
			SampleRate: 24000,
			Channels:   1,
			FrameSize:  480,
			Codec:      "PCM16",
		},

		Data: data,
	}, nil
}
