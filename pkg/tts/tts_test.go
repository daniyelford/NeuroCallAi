package tts

import (
	"context"
	"testing"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func TestLocalTTSSynthesize(t *testing.T) {
	tts := NewLocalTTS()

	audio, err := tts.Synthesize(
		context.Background(),
		"hello",
	)

	if err != nil {
		t.Fatal(err)
	}

	if len(audio.Data) == 0 {
		t.Fatal("expected audio data")
	}

	if audio.Format.Codec != "PCM16" {
		t.Fatalf(
			"expected PCM16 codec, got %q",
			audio.Format.Codec,
		)
	}

	if audio.Format.SampleRate != 8000 {
		t.Fatalf(
			"expected 8000 Hz, got %d",
			audio.Format.SampleRate,
		)
	}

	if audio.Format.Channels != 1 {
		t.Fatalf(
			"expected mono audio, got %d channels",
			audio.Format.Channels,
		)
	}
}

func TestLocalTTSCancelledContext(t *testing.T) {
	tts := NewLocalTTS()

	ctx, cancel := context.WithCancel(
		context.Background(),
	)

	cancel()

	_, err := tts.Synthesize(ctx, "hello")

	if err != context.Canceled {
		t.Fatalf(
			"expected context.Canceled, got %v",
			err,
		)
	}
}

func TestLocalTTSImplementsInterface(t *testing.T) {
	var _ neurocall.TTS = (*LocalTTS)(nil)
}
