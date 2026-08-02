package audio

import (
	"context"
	"testing"
	"time"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

func TestPipeline(t *testing.T) {

	buffer := NewBuffer(4)

	sink := NewMemorySink()

	pipeline := NewPipeline(
		buffer,
		Passthrough{},
		sink,
	)

	ctx, cancel := context.WithCancel(
		context.Background(),
	)
	defer cancel()

	done := make(chan error, 1)

	go func() {
		done <- pipeline.Run(ctx)
	}()

	frame := neurocall.AudioFrame{
		Data:      []byte{1, 2, 3, 4},
		Format:    neurocall.PCM16Mono16K,
		Timestamp: time.Now(),
	}

	if err := buffer.Write(ctx, frame); err != nil {
		t.Fatal(err)
	}

	time.Sleep(10 * time.Millisecond)

	cancel()

	<-done

	frames := sink.Frames()

	if len(frames) != 1 {
		t.Fatalf(
			"expected 1 frame, got %d",
			len(frames),
		)
	}
}
