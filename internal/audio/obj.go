package audio

import (
	"sync"
	"time"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

type Passthrough struct{}
type Buffer struct {
	ch chan neurocall.AudioFrame

	closeOnce sync.Once
}
type Pipeline struct {
	source     neurocall.AudioSource
	processors []neurocall.AudioProcessor
	sink       neurocall.AudioSink
}
type MemorySink struct {
	mu     sync.Mutex
	frames []neurocall.AudioFrame
}
type AudioChunk struct {
	Data      []byte
	Format    AudioFormat
	Timestamp time.Duration
}
type Engine struct {
	mu sync.Mutex

	pipelines []*Pipeline

	running bool
}
type RingBuffer struct {
	mu sync.Mutex

	frames []neurocall.AudioFrame

	head int
	tail int
	size int
}
type LinearResampler struct {
	targetRate int
}
type EnergyVAD struct {
	Threshold float64
}
type SpeechDetector struct {
	vad *EnergyVAD

	active bool

	frames []neurocall.AudioFrame

	start time.Duration

	silence time.Duration
}
