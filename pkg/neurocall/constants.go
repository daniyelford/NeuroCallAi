package neurocall

import "errors"

const (
	Created State = iota
	Initializing
	Starting
	Running
	Stopping
	Stopped
)
const (
	Version2 = 2

	HeaderSize = 12

	PayloadPCMU uint8 = 0
	PayloadPCMA uint8 = 8
)

var PCM16Mono16K = AudioFormat{
	SampleRate:    16000,
	Channels:      1,
	BitsPerSample: 16,
}
var (
	ErrInvalidDependencyType = errors.New(
		"invalid dependency type",
	)
	ErrInvalidState = errors.New(
		"invalid application state",
	)
	ErrAudioClosed = errors.New(
		"audio source is closed",
	)

	ErrInvalidAudioFormat = errors.New(
		"invalid audio format",
	)

	ErrEmptyAudioFrame = errors.New(
		"audio frame is empty",
	)
)
