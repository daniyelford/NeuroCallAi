package core

import (
	"context"
	"fmt"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewAudioProcessor(
	vad neurocall.VAD,
	segmenter neurocall.SpeechSegmenter,
	worker *STTWorker,
	bus *EventBus,
) (*AudioProcessor, error) {

	if vad == nil {
		return nil, fmt.Errorf("VAD is nil")
	}

	if segmenter == nil {
		return nil, fmt.Errorf("speech segmenter is nil")
	}

	if worker == nil {
		return nil, fmt.Errorf("STT worker is nil")
	}

	if bus == nil {
		return nil, fmt.Errorf("event bus is nil")
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &AudioProcessor{
		vad:       vad,
		segmenter: segmenter,
		worker:    worker,
		bus:       bus,
		ctx:       ctx,
		cancel:    cancel,
	}, nil
}
func (p *AudioProcessor) Start() error {
	if p == nil {
		return neurocall.ErrCallClosed
	}

	p.mu.Lock()

	if p.closed {
		p.mu.Unlock()
		return neurocall.ErrCallClosed
	}

	if p.running {
		p.mu.Unlock()
		return nil
	}

	p.running = true

	worker := p.worker

	p.mu.Unlock()

	if worker != nil {
		if err := worker.Start(); err != nil {
			p.mu.Lock()
			p.running = false
			p.mu.Unlock()

			return err
		}
	}

	return nil
}
func (p *AudioProcessor) Stop() error {
	if p == nil {
		return nil
	}

	p.mu.Lock()

	if p.closed {
		p.mu.Unlock()
		return nil
	}

	p.closed = true
	p.running = false

	cancel := p.cancel
	worker := p.worker

	p.mu.Unlock()

	if cancel != nil {
		cancel()
	}

	if worker != nil {
		worker.Stop()
	}

	return nil
}
func (p *AudioProcessor) Running() bool {
	if p == nil {
		return false
	}

	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.running && !p.closed
}
func (p *AudioProcessor) Closed() bool {
	if p == nil {
		return true
	}

	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.closed
}
func (p *AudioProcessor) Context() context.Context {
	if p == nil {
		return context.Background()
	}

	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.ctx
}
func (p *AudioProcessor) SetVAD(
	vad neurocall.VAD,
) error {

	if vad == nil {
		return fmt.Errorf("VAD is nil")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return neurocall.ErrCallClosed
	}

	p.vad = vad

	return nil
}
func (p *AudioProcessor) SetSegmenter(
	segmenter neurocall.SpeechSegmenter,
) error {

	if segmenter == nil {
		return fmt.Errorf("speech segmenter is nil")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return neurocall.ErrCallClosed
	}

	p.segmenter = segmenter

	return nil
}
func (p *AudioProcessor) SetWorker(
	worker *STTWorker,
) error {

	if worker == nil {
		return fmt.Errorf("STT worker is nil")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return neurocall.ErrCallClosed
	}

	p.worker = worker

	return nil
}
func (p *AudioProcessor) VAD() neurocall.VAD {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.vad
}
func (p *AudioProcessor) Segmenter() neurocall.SpeechSegmenter {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.segmenter
}
func (p *AudioProcessor) Worker() *STTWorker {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.worker
}
func (p *AudioProcessor) ProcessFrame(
	frame neurocall.AudioFrame,
) error {

	if p == nil {
		return neurocall.ErrCallClosed
	}

	if len(frame.Data) == 0 {
		return neurocall.ErrInvalidAudio
	}

	if frame.SampleRate <= 0 {
		return neurocall.ErrInvalidAudio
	}

	if frame.Channels <= 0 {
		return neurocall.ErrInvalidAudio
	}

	p.mu.RLock()

	if p.closed || !p.running {
		p.mu.RUnlock()
		return neurocall.ErrCallClosed
	}

	ctx := p.ctx
	vad := p.vad
	segmenter := p.segmenter
	worker := p.worker

	p.mu.RUnlock()

	if vad == nil {
		return fmt.Errorf("VAD is not configured")
	}

	if segmenter == nil {
		return fmt.Errorf("speech segmenter is not configured")
	}

	if worker == nil {
		return fmt.Errorf("STT worker is not configured")
	}

	speech, err := vad.Process(ctx, frame)
	if err != nil {
		p.publishVADError(err)
		return err
	}

	if !speech {
		return nil
	}

	segments, err := segmenter.Process(ctx, frame)
	if err != nil {
		p.publishSegmenterError(err)
		return err
	}

	for _, segment := range segments {
		if len(segment.Data) == 0 {
			continue
		}

		if err := worker.Push(segment); err != nil {
			return err
		}

		p.publishSegment(segment)
	}

	return nil
}
func (p *AudioProcessor) Flush() error {
	if p == nil {
		return neurocall.ErrCallClosed
	}

	p.mu.RLock()

	if p.closed {
		p.mu.RUnlock()
		return neurocall.ErrCallClosed
	}

	ctx := p.ctx
	segmenter := p.segmenter
	worker := p.worker

	p.mu.RUnlock()

	if segmenter == nil {
		return fmt.Errorf("speech segmenter is not configured")
	}

	if worker == nil {
		return fmt.Errorf("STT worker is not configured")
	}

	segments, err := segmenter.Flush(ctx)
	if err != nil {
		p.publishSegmenterError(err)
		return err
	}

	for _, segment := range segments {
		if len(segment.Data) == 0 {
			continue
		}

		if err := worker.Push(segment); err != nil {
			return err
		}

		p.publishSegment(segment)
	}

	return nil
}
func (p *AudioProcessor) ResetSegmenter() error {
	if p == nil {
		return neurocall.ErrCallClosed
	}

	p.mu.RLock()

	if p.closed {
		p.mu.RUnlock()
		return neurocall.ErrCallClosed
	}

	segmenter := p.segmenter

	p.mu.RUnlock()

	if segmenter == nil {
		return fmt.Errorf("speech segmenter is not configured")
	}

	segmenter.Reset()

	return nil
}
func (p *AudioProcessor) publishSegment(
	segment neurocall.AudioSegment,
) {
	if p == nil || p.bus == nil {
		return
	}
	p.bus.Publish(Event{
		Name: EventAudioSegment,
		Data: segment,
	})
}
func (p *AudioProcessor) publishVADError(
	err error,
) {
	if p == nil || p.bus == nil {
		return
	}
	p.bus.Publish(Event{
		Name: EventAudioProcessorError,
		Data: err,
	})
}
func (p *AudioProcessor) publishSegmenterError(
	err error,
) {
	if p == nil || p.bus == nil {
		return
	}

	p.bus.Publish(Event{
		Name: EventAudioProcessorError,
		Data: err,
	})
}
