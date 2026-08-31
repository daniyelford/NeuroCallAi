package core

import (
	"context"
	"fmt"
	"time"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewAudioSegmenter(
	vad neurocall.VAD,
	cfg AudioConfig,
) *AudioSegmenter {
	frameDuration := time.Second
	if cfg.SampleRate > 0 {
		frameDuration =
			time.Duration(
				float64(cfg.FrameSize) /
					float64(cfg.SampleRate) *
					float64(time.Second),
			)
	}

	return &AudioSegmenter{
		vad: vad,

		buffer: make(
			[]int16,
			0,
			cfg.SampleRate*5,
		),

		maxFrames: 250,

		frameDuration: frameDuration,

		sampleRate: cfg.SampleRate,

		channels: cfg.Channels,
	}
}
func (s *AudioSegmenter) Process(
	ctx context.Context,
	frame neurocall.AudioFrame,
) ([]neurocall.AudioSegment, error) {

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.vad == nil {
		return nil, fmt.Errorf(
			"VAD is not configured",
		)
	}

	speech, err :=
		s.vad.Process(ctx, frame)

	if err != nil {
		return nil, err
	}

	if speech {
		return s.processSpeech(frame), nil
	}

	return s.processSilence(frame), nil
}
func (s *AudioSegmenter) processSpeech(
	frame neurocall.AudioFrame,
) []neurocall.AudioSegment {
	if !s.active {
		s.active = true
		s.start = time.Duration(
			frame.Timestamp,
		) * time.Millisecond
		s.frameCount = 0
	}
	s.silenceFrames = 0
	s.buffer = append(
		s.buffer,
		frame.Data...,
	)
	s.frameCount++
	if s.maxFrames > 0 && s.frameCount >= s.maxFrames {
		return s.flushLocked(true)
	}
	s.end += s.frameDuration
	return nil
}
func (s *AudioSegmenter) processSilence(
	_ neurocall.AudioFrame,
) []neurocall.AudioSegment {
	if !s.active {
		return nil
	}
	s.silenceFrames++
	s.end += s.frameDuration
	// ~200ms silence
	if s.silenceFrames >= 10 {
		return s.flushLocked(true)
	}
	return nil
}
func (s *AudioSegmenter) flushLocked(
	final bool,
) []neurocall.AudioSegment {

	if len(s.buffer) == 0 {
		s.resetLocked()
		return nil
	}

	data := make(
		[]int16,
		len(s.buffer),
	)

	copy(data, s.buffer)

	segment := neurocall.AudioSegment{
		Data:       data,
		SampleRate: s.sampleRate,
		Channels:   s.channels,
		Start:      s.start,
		End:        s.end,
		Final:      final,
	}
	s.resetLocked()
	return []neurocall.AudioSegment{
		segment,
	}
}
func (s *AudioSegmenter) resetLocked() {
	s.buffer = s.buffer[:0]

	s.active = false

	s.silenceFrames = 0

	s.start = 0
	s.end = 0
}
func (s *AudioSegmenter) Flush(
	ctx context.Context,
) ([]neurocall.AudioSegment, error) {

	select {
	case <-ctx.Done():
		return nil, ctx.Err()

	default:
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	return s.flushLocked(true), nil
}
func (s *AudioSegmenter) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.frameCount = 0
	s.resetLocked()
}
