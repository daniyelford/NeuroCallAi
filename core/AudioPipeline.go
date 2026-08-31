package core

import (
	"context"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewAudioPipeline(cfg AudioConfig) *AudioPipeline {
	ctx, cancel := context.WithCancel(context.Background())
	return &AudioPipeline{
		input:  make(chan []int16, cfg.BufferSize),
		output: make(chan []int16, cfg.BufferSize),
		ctx:    ctx,
		cancel: cancel,
	}
}
func (p *AudioPipeline) Start() {
	if p == nil {
		return
	}
	p.mu.Lock()
	if p.closed || p.running {
		p.mu.Unlock()
		return
	}
	p.running = true
	p.mu.Unlock()
	go p.run()
}
func (p *AudioPipeline) Decode(data []byte) ([]int16, error) {
	p.mu.RLock()
	decoder := p.decoder
	p.mu.RUnlock()
	if decoder == nil {
		return nil, neurocall.ErrInvalidAudio
	}
	if len(data) == 0 {
		return nil,
			neurocall.ErrInvalidAudio
	}
	pcm := decoder.Decode(data)
	if len(pcm) == 0 {
		return nil, neurocall.ErrInvalidAudio
	}
	return pcm, nil
}
func (p *AudioPipeline) Encode(pcm []int16) ([]byte, error) {
	p.mu.RLock()
	encoder := p.encoder
	p.mu.RUnlock()
	if encoder == nil {
		return nil, neurocall.ErrInvalidAudio
	}
	if len(pcm) == 0 {
		return nil, neurocall.ErrInvalidAudio
	}
	payload := encoder.Encode(pcm)
	if len(payload) == 0 {
		return nil, neurocall.ErrInvalidAudio
	}
	return payload, nil
	// return encoder.Encode(pcm), nil
}
func (p *AudioPipeline) Close() error {
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
	p.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	return nil
}
func (p *AudioPipeline) SetEncoder(encoder neurocall.Encoder) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.encoder = encoder
}
func (p *AudioPipeline) SetDecoder(decoder neurocall.Decoder) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.decoder = decoder
}
func (p *AudioPipeline) SetResampler(resampler neurocall.Resampler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.resampler = resampler
}
func (p *AudioPipeline) Resample(input []int16, fromRate int, toRate int) []int16 {
	p.mu.RLock()
	resampler := p.resampler
	p.mu.RUnlock()
	if resampler == nil {
		return input
	}
	if fromRate == toRate {
		return input
	}
	return resampler.Resample(input, fromRate, toRate)
}
func (p *AudioPipeline) Encoder() neurocall.Encoder {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.encoder
}
func (p *AudioPipeline) Decoder() neurocall.Decoder {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.decoder
}
func (p *AudioPipeline) Resampler() neurocall.Resampler {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.resampler
}
func (p *AudioPipeline) Push(audio []int16) error {
	if p == nil {
		return neurocall.ErrCallClosed
	}
	if len(audio) == 0 {
		return neurocall.ErrInvalidAudio
	}
	p.mu.RLock()
	if p.closed || !p.running {
		p.mu.RUnlock()
		return neurocall.ErrCallClosed
	}
	input := p.input
	ctx := p.ctx
	p.mu.RUnlock()
	select {
	case <-ctx.Done():
		return neurocall.ErrCallClosed
	case input <- audio:
		return nil
	}
}
func (p *AudioPipeline) Pull() ([]int16, error) {
	if p == nil {
		return nil, neurocall.ErrCallClosed
	}

	select {
	case <-p.ctx.Done():
		return nil, neurocall.ErrCallClosed

	case audio, ok := <-p.output:
		if !ok {
			return nil, neurocall.ErrCallClosed
		}

		if len(audio) == 0 {
			return nil, neurocall.ErrInvalidAudio
		}

		return audio, nil
	}
}
func (p *AudioPipeline) run() {
	defer func() {
		p.mu.Lock()
		p.running = false
		p.mu.Unlock()
	}()
	for {
		select {
		case <-p.ctx.Done():
			return
		case pcm, ok := <-p.input:
			if !ok {
				return
			}
			if len(pcm) == 0 {
				continue
			}
			p.mu.RLock()
			encoder := p.encoder
			decoder := p.decoder
			p.mu.RUnlock()
			if encoder == nil || decoder == nil {
				continue
			}
			payload := encoder.Encode(pcm)
			if len(payload) == 0 {
				continue
			}
			decoded := decoder.Decode(payload)
			if len(decoded) == 0 {
				continue
			}
			select {
			case p.output <- decoded:
			case <-p.ctx.Done():
				return
			}
		}
	}
}
