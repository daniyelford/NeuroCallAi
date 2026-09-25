package tts

import (
	"context"
	"encoding/binary"
	"math"
	"strings"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewLocalTTS() *LocalTTS {
	return &LocalTTS{
		SampleRate: DefaultSampleRate,
		Channels:   DefaultChannels,
	}
}
func (t *LocalTTS) Synthesize(
	ctx context.Context,
	text string,
) (neurocall.AudioStreamData, error) {

	if t == nil {
		return neurocall.AudioStreamData{}, ErrInvalidLocalTTS
	}

	text = strings.TrimSpace(text)

	if text == "" {
		return neurocall.AudioStreamData{}, ErrEmptyText
	}

	if ctx == nil {
		ctx = context.Background()
	}

	select {
	case <-ctx.Done():
		return neurocall.AudioStreamData{}, ctx.Err()
	default:
	}

	sampleRate := t.SampleRate
	if sampleRate <= 0 {
		sampleRate = DefaultSampleRate
	}

	channels := t.Channels
	if channels <= 0 {
		channels = DefaultChannels
	}

	pcm := synthesizeText(text, sampleRate)

	select {
	case <-ctx.Done():
		return neurocall.AudioStreamData{}, ctx.Err()
	default:
	}

	return neurocall.AudioStreamData{
		Format: neurocall.AudioFormat{
			SampleRate: sampleRate,
			Channels:   channels,
			FrameSize:  DefaultFrameSize,
			Codec:      "PCM16",
		},
		Data: pcm,
	}, nil
}
func synthesizeText(text string, sampleRate int) []byte {
	const (
		charDuration = 0.05
		frequency    = 220.0
		amplitude    = 3000.0
	)

	samplesPerChar := int(float64(sampleRate) * charDuration)

	totalSamples := len([]rune(text)) * samplesPerChar

	pcm := make([]byte, totalSamples*2)

	index := 0

	for _, char := range text {
		_ = char

		for i := 0; i < samplesPerChar; i++ {
			t := float64(i) / float64(sampleRate)

			envelope := 1.0

			if i < sampleRate/100 {
				envelope = float64(i) / float64(sampleRate/100)
			}

			remaining := samplesPerChar - i

			if remaining < sampleRate/100 {
				envelope = float64(remaining) / float64(sampleRate/100)
			}

			sample := amplitude *
				envelope *
				math.Sin(2*math.Pi*frequency*t)

			value := int16(sample)

			binary.LittleEndian.PutUint16(
				pcm[index:index+2],
				uint16(value),
			)

			index += 2
		}
	}

	return pcm
}
