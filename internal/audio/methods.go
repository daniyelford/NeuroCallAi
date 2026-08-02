package audio

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

func (b *Buffer) Close() {

	b.closeOnce.Do(func() {
		close(b.ch)
	})
}
func NewBuffer(size int) *Buffer {

	if size <= 0 {
		size = 32
	}

	return &Buffer{
		ch: make(
			chan neurocall.AudioFrame,
			size,
		),
	}
}

func (b *Buffer) Write(
	ctx context.Context,
	frame neurocall.AudioFrame,
) error {

	if len(frame.Data) == 0 {
		return neurocall.ErrEmptyAudioFrame
	}

	select {

	case b.ch <- frame:
		return nil

	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *Buffer) Read(
	ctx context.Context,
) (neurocall.AudioFrame, error) {

	select {

	case frame, ok := <-b.ch:

		if !ok {
			return neurocall.AudioFrame{},
				neurocall.ErrAudioClosed
		}

		return frame, nil

	case <-ctx.Done():
		return neurocall.AudioFrame{},
			ctx.Err()
	}
}

func (Passthrough) Process(
	_ context.Context,
	frame neurocall.AudioFrame,
) (neurocall.AudioFrame, error) {

	return frame, nil
}

func NewMemorySink() *MemorySink {
	return &MemorySink{}
}

func (s *MemorySink) Write(
	_ context.Context,
	frame neurocall.AudioFrame,
) error {

	s.mu.Lock()
	defer s.mu.Unlock()

	s.frames = append(
		s.frames,
		frame,
	)

	return nil
}

func (s *MemorySink) Frames() []neurocall.AudioFrame {

	s.mu.Lock()
	defer s.mu.Unlock()

	return append(
		[]neurocall.AudioFrame(nil),
		s.frames...,
	)
}

func NewEngine() *Engine {
	return &Engine{}
}

func (e *Engine) AddPipeline(
	pipeline *Pipeline,
) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.pipelines = append(
		e.pipelines,
		pipeline,
	)
}

func (e *Engine) Run(
	ctx context.Context,
) error {

	e.mu.Lock()

	if e.running {
		e.mu.Unlock()

		return nil
	}

	e.running = true

	pipelines := append(
		[]*Pipeline(nil),
		e.pipelines...,
	)

	e.mu.Unlock()

	var wg sync.WaitGroup

	errCh := make(chan error, len(pipelines))

	for _, pipeline := range pipelines {

		wg.Add(1)

		go func(p *Pipeline) {

			defer wg.Done()

			if err := p.Run(ctx); err != nil {
				errCh <- err
			}

		}(pipeline)
	}

	wg.Wait()

	close(errCh)

	for err := range errCh {
		if err != nil {
			return err
		}
	}

	return nil
}

// func NewPipeline(
// 	source neurocall.AudioSource,
// 	processor neurocall.AudioProcessor,
// 	sink neurocall.AudioSink,
// ) *Pipeline {

// 	return &Pipeline{
// 		source:    source,
// 		processor: processor,
// 		sink:      sink,
// 	}
// }

// func (p *Pipeline) Run(
// 	ctx context.Context,
// ) error {

// 	for {

// 		frame, err := p.source.Read(ctx)

// 		if err != nil {
// 			return err
// 		}

// 		frame, err = p.processor.Process(
// 			ctx,
// 			frame,
// 		)

// 		if err != nil {
// 			return err
// 		}

// 		if err := p.sink.Write(
// 			ctx,
// 			frame,
// 		); err != nil {
// 			return err
// 		}
// 	}
// }

func FrameSize(
	format neurocall.AudioFormat,
	duration time.Duration,
) int {
	return format.BytesPerFrame(duration)
}

func NewEmptyFrame(
	format neurocall.AudioFormat,
	timestamp time.Duration,
) neurocall.AudioFrame {

	size := FrameSize(
		format,
		DefaultFrameDuration,
	)

	return neurocall.AudioFrame{
		Data:      make([]byte, size),
		Format:    format,
		Timestamp: timestamp,
	}
}

func NewRingBuffer(capacity int) *RingBuffer {
	if capacity <= 0 {
		capacity = 50
	}

	return &RingBuffer{
		frames: make(
			[]neurocall.AudioFrame,
			capacity,
		),
	}
}

func (r *RingBuffer) Write(
	frame neurocall.AudioFrame,
) error {

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.size == len(r.frames) {
		return ErrBufferFull
	}

	r.frames[r.tail] = frame

	r.tail = (r.tail + 1) % len(r.frames)
	r.size++

	return nil
}

func (r *RingBuffer) Read() (
	neurocall.AudioFrame,
	error,
) {

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.size == 0 {
		return neurocall.AudioFrame{},
			ErrBufferEmpty
	}

	frame := r.frames[r.head]

	r.frames[r.head] = neurocall.AudioFrame{}

	r.head = (r.head + 1) % len(r.frames)
	r.size--

	return frame, nil
}

func (r *RingBuffer) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.size
}

func NewLinearResampler(
	targetRate int,
) *LinearResampler {

	return &LinearResampler{
		targetRate: targetRate,
	}
}

func (r *LinearResampler) Process(
	_ context.Context,
	frame neurocall.AudioFrame,
) (neurocall.AudioFrame, error) {
	if frame.Format.BitsPerSample != 16 {
		return neurocall.AudioFrame{},
			fmt.Errorf(
				"unsupported bits per sample: %d",
				frame.Format.BitsPerSample,
			)
	}

	if frame.Format.Channels != 1 {
		return neurocall.AudioFrame{},
			fmt.Errorf(
				"only mono audio is supported",
			)
	}

	if frame.Format.SampleRate == r.targetRate {
		return frame, nil
	}

	if len(frame.Data)%2 != 0 {
		return neurocall.AudioFrame{},
			fmt.Errorf(
				"invalid PCM16 byte length",
			)
	}

	inputSamples := len(frame.Data) / 2

	if inputSamples == 0 {
		return frame, nil
	}

	outputSamples := int(
		float64(inputSamples) *
			float64(r.targetRate) /
			float64(frame.Format.SampleRate),
	)

	output := make(
		[]byte,
		outputSamples*2,
	)

	for i := 0; i < outputSamples; i++ {

		position :=
			float64(i) *
				float64(inputSamples-1) /
				float64(outputSamples-1)

		left := int(position)
		right := left + 1

		if right >= inputSamples {
			right = inputSamples - 1
		}

		fraction := position -
			float64(left)

		leftValue := int16(
			binary.LittleEndian.Uint16(
				frame.Data[left*2:],
			),
		)

		rightValue := int16(
			binary.LittleEndian.Uint16(
				frame.Data[right*2:],
			),
		)

		value := float64(leftValue) +
			(float64(rightValue)-
				float64(leftValue))*fraction

		binary.LittleEndian.PutUint16(
			output[i*2:],
			uint16(int16(value)),
		)
	}

	frame.Data = output
	frame.Format.SampleRate = r.targetRate

	return frame, nil
}

func NewPipeline(
	source neurocall.AudioSource,
	processors []neurocall.AudioProcessor,
	sink neurocall.AudioSink,
) *Pipeline {

	return &Pipeline{
		source:     source,
		processors: processors,
		sink:       sink,
	}
}

func (p *Pipeline) Run(
	ctx context.Context,
) error {

	for {

		frame, err := p.source.Read(ctx)

		if err != nil {
			return err
		}

		for _, processor := range p.processors {

			frame, err = processor.Process(
				ctx,
				frame,
			)

			if err != nil {
				return err
			}
		}

		if err := p.sink.Write(
			ctx,
			frame,
		); err != nil {
			return err
		}
	}
}

func RMS(frame neurocall.AudioFrame) float64 {

	if len(frame.Data) < 2 {
		return 0
	}

	var sum float64

	count := len(frame.Data) / 2

	for i := 0; i < count; i++ {

		sample := int16(
			binary.LittleEndian.Uint16(
				frame.Data[i*2:],
			),
		)

		normalized :=
			float64(sample) / 32768.0

		sum += normalized * normalized
	}

	return math.Sqrt(sum / float64(count))
}
func NewEnergyVAD(
	threshold float64,
) *EnergyVAD {

	if threshold <= 0 {
		threshold = 0.015
	}

	return &EnergyVAD{
		Threshold: threshold,
	}
}

func (v *EnergyVAD) Process(
	_ context.Context,
	frame neurocall.AudioFrame,
) (neurocall.AudioFrame, error) {

	return frame, nil
}

func (v *EnergyVAD) IsSpeech(
	frame neurocall.AudioFrame,
) bool {

	return RMS(frame) >= v.Threshold
}
func NewSpeechDetector(
	vad *EnergyVAD,
) *SpeechDetector {

	return &SpeechDetector{
		vad: vad,
	}
}

func (d *SpeechDetector) Push(
	frame neurocall.AudioFrame,
) *neurocall.SpeechSegment {

	if d.vad.IsSpeech(frame) {

		if !d.active {
			d.active = true
			d.start = frame.Timestamp
		}

		d.silence = 0

		d.frames = append(
			d.frames,
			frame,
		)

		return nil
	}

	if !d.active {
		return nil
	}

	d.silence += frame.Duration()

	d.frames = append(
		d.frames,
		frame,
	)

	if d.silence < 300*time.Millisecond {
		return nil
	}

	segment := &neurocall.SpeechSegment{
		Frames: d.frames,
		Start:  d.start,
		End:    frame.Timestamp + frame.Duration(),
	}

	d.active = false
	d.frames = nil
	d.silence = 0

	return segment
}
