package core

import (
	"context"
	"fmt"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewSTTWorker(
	engine *STTEngine,
	bus *EventBus,
	callID string,
	bufferSize int,
) (*STTWorker, error) {

	if engine == nil {
		return nil, fmt.Errorf("STT engine is nil")
	}

	if bus == nil {
		return nil, fmt.Errorf("event bus is nil")
	}

	if callID == "" {
		return nil, fmt.Errorf("call ID is empty")
	}

	if bufferSize <= 0 {
		bufferSize = 16
	}

	ctx, cancel := context.WithCancel(
		context.Background(),
	)

	return &STTWorker{
		engine: engine,
		bus:    bus,
		callID: callID,
		input:  make(chan neurocall.AudioSegment, bufferSize),
		ctx:    ctx,
		cancel: cancel,
	}, nil
}
func (w *STTWorker) SetStreamingSTT(
	stt neurocall.StreamingSTT,
) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.streaming = stt
}
func (w *STTWorker) Push(segment neurocall.AudioSegment) error {
	if len(segment.Data) == 0 {
		return neurocall.ErrInvalidAudio
	}
	w.mu.Lock()

	if w.stopped {
		w.mu.Unlock()
		return neurocall.ErrCallClosed
	}

	input := w.input
	ctx := w.ctx

	w.mu.Unlock()

	select {

	case <-ctx.Done():
		return neurocall.ErrCallClosed

	case input <- segment:
		return nil
	}
}
func (w *STTWorker) Start() error {
	w.mu.Lock()

	if w.stopped {
		w.mu.Unlock()
		return neurocall.ErrCallClosed
	}

	if w.running {
		w.mu.Unlock()
		return nil
	}

	w.running = true
	streaming := w.streaming
	ctx := w.ctx

	w.mu.Unlock()

	if streaming != nil {
		if err := streaming.Start(ctx); err != nil {
			w.mu.Lock()
			w.running = false
			w.mu.Unlock()
			return err
		}

		go w.runStreaming(streaming)
	}

	go w.run()

	return nil
}
func (w *STTWorker) runStreaming(streaming neurocall.StreamingSTT) {
	events := streaming.Events()

	for {
		select {
		case <-w.ctx.Done():
			return
		case event, ok := <-events:
			if !ok {
				return
			}
			w.bus.Publish(Event{
				Name: EventSTTEvent,
				Data: STTEventData{
					CallID: w.callID,
					Event:  event,
				},
			})
		}
	}
}
func (w *STTWorker) run() {
	for {
		select {

		case <-w.ctx.Done():
			return

		case segment, ok := <-w.input:
			if !ok {
				return
			}

			transcript, err := w.engine.Transcribe(
				w.ctx,
				segment,
			)

			if err != nil {
				w.bus.Publish(Event{
					Name: EventSTTError,
					Data: STTErrorEvent{
						CallID: w.callID,
						Err:    err,
					},
				})

				continue
			}

			w.bus.Publish(Event{
				Name: EventTranscript,
				Data: TranscriptEvent{
					CallID:     w.callID,
					Transcript: transcript,
				},
			})
		}
	}
}
func (w *STTWorker) Stop() {
	w.mu.Lock()

	if w.stopped {
		w.mu.Unlock()
		return
	}

	w.stopped = true
	w.running = false

	cancel := w.cancel
	streaming := w.streaming

	w.mu.Unlock()

	if cancel != nil {
		cancel()
	}

	if streaming != nil {
		_ = streaming.Close()
	}
}
