package tts

import "errors"

var (
	ErrInvalidLocalTTS = errors.New("invalid local tts")
	ErrEmptyText       = errors.New("empty text")
)

const (
	DefaultSampleRate = 8000
	DefaultChannels   = 1
	DefaultFrameSize  = 160
)
