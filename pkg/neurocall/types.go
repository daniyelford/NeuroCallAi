package neurocall

import (
	"context"
	"time"
)

// Codec represents an audio codec supported by the voice runtime.
// IncomingCall represents an incoming telephony call.
// Call represents a generic call abstraction.
// AudioStream represents bidirectional call audio.
// Encoder converts PCM audio into a codec payload.
// Decoder converts codec payload into PCM audio.
// Resampler converts audio between sample rates.
// Tool represents a callable AI tool.
// ToolRegistry provides access to registered AI tools.
// Message represents an LLM conversation message.
// Memory stores conversation state.
// VAD detects whether an audio frame contains speech.
// SpeechSegmenter converts audio frames into speech segments.

type Codec struct {
	Name        string
	PayloadType uint8
	ClockRate   int
	Channels    int
}
type IncomingCall struct {
	ID   string
	From string
	To   string
}
type Call interface {
	ID() string
	Answer() error
	Hangup() error
}
type AudioStream interface {
	Read() ([]int16, error)
	Write([]int16) error
	Close() error
}
type Encoder interface {
	Encode(pcm []int16) []byte
}
type Decoder interface {
	Decode(data []byte) []int16
}
type Resampler interface {
	Resample(
		input []int16,
		fromRate int,
		toRate int,
	) []int16
}
type Tool interface {
	Name() string
	Description() string
	Call(args map[string]any) (any, error)
}
type ToolRegistry interface {
	Register(tool Tool) error
	Get(name string) (Tool, bool)
	List() []Tool
}
type LLM interface {
	Chat(
		messages []Message,
	) (Message, error)
}
type Message struct {
	Role    string
	Content string
}
type Memory interface {
	Get(key string) (any, bool)
	Set(key string, value any)
	Delete(key string)
	Clear()
}
type Transcript struct {
	Text       string
	Confidence float64
	Language   string
	Start      int64
	End        int64
}
type StreamingSTT interface {
	Start(ctx context.Context) error
	Write(
		ctx context.Context,
		frame AudioFrame,
	) error
	Events() <-chan STTEvent
	Close() error
}
type STT interface {
	Transcribe(
		ctx context.Context,
		segment AudioSegment,
	) (Transcript, error)
}
type STTEvent struct {
	Type       STTEventType
	Text       string
	Final      bool
	Confidence float64
	Start      time.Duration
	End        time.Duration
}
type STTEventType uint8
type TTS interface {
	Synthesize(
		ctx context.Context,
		text string,
	) (AudioStreamData, error)
}
type AudioStreamData struct {
	Format AudioFormat
	Data   []byte
}
type SpeechSegment struct {
	Text       string
	Start      time.Duration
	End        time.Duration
	Confidence float32
	Final      bool
}
type AudioFrame struct {
	Data       []int16
	Timestamp  uint32
	SampleRate int
	Channels   int
}
type AudioFormat struct {
	SampleRate int
	Channels   int
	FrameSize  int
	Codec      string
}
type AudioSegment struct {
	Data       []int16
	SampleRate int
	Channels   int
	Start      time.Duration
	End        time.Duration
	Final      bool
}
type VAD interface {
	Process(
		ctx context.Context,
		frame AudioFrame,
	) (bool, error)
}
type SpeechSegmenter interface {
	Process(
		ctx context.Context,
		frame AudioFrame,
	) ([]AudioSegment, error)

	Flush(
		ctx context.Context,
	) ([]AudioSegment, error)

	Reset()
}
