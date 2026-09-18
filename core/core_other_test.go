package core

import (
	"context"
	"math"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
	"github.com/daniyelford/NeuroCallAi/pkg/openaipkg"
)

type testVAD struct {
	mu     sync.Mutex
	result bool
	calls  int
	err    error
}
type testSegmenter struct {
	mu           sync.Mutex
	segments     []neurocall.AudioSegment
	processCalls int
	flushCalls   int
	resetCalls   int
	err          error
}
type testSTTEngine struct{}
type testClosableResource struct {
	mu     sync.Mutex
	closed bool
}
type testLLM struct {
	response neurocall.Message
	err      error
	calls    int
	messages [][]neurocall.Message
}
type testTTS struct {
	calls int
	texts []string
	audio neurocall.AudioStreamData
	err   error
}
type testTTSFail struct {
	data  neurocall.AudioStreamData
	err   error
	calls int
}
type integrationTestLLM struct {
	calls    int
	response neurocall.Message
}
type integrationTestTTS struct {
	calls int
	text  string
	data  neurocall.AudioStreamData
}
type testSTT struct {
	transcribe func(
		context.Context,
		neurocall.AudioSegment,
	) (neurocall.Transcript, error)
}
type integrationSTT struct {
	called chan neurocall.AudioSegment
}
type testPlugin struct {
	name string
}
type testTool struct {
	name string
}
type pipelineTestEncoder struct {
	output []byte
}
type pipelineTestDecoder struct {
	output []int16
}
type pipelineTestResampler struct {
	output   []int16
	called   bool
	fromRate int
	toRate   int
}
type audioProcessorTestVAD struct {
	speech bool
	err    error
	calls  int
}
type audioProcessorTestSegmenter struct {
	segments     []neurocall.AudioSegment
	processErr   error
	flushErr     error
	processCalls int
	flushCalls   int
	resetCalls   int
}

func newAudioProcessorTestWorker(
	t *testing.T,
	bus *EventBus,
) *STTWorker {
	t.Helper()

	engine := NewSTTEngine(NewFakeSTT())

	worker, err := NewSTTWorker(
		engine,
		bus,
		"audio-processor-test-call",
		8,
	)
	if err != nil {
		t.Fatalf("NewSTTWorker() error = %v", err)
	}

	return worker
}
func newAudioProcessorTestProcessor(
	t *testing.T,
	vad neurocall.VAD,
	segmenter neurocall.SpeechSegmenter,
) (*AudioProcessor, *EventBus, *STTWorker) {
	t.Helper()

	bus := NewEventBus()

	worker := newAudioProcessorTestWorker(t, bus)

	processor, err := NewAudioProcessor(
		vad,
		segmenter,
		worker,
		bus,
	)
	if err != nil {
		t.Fatalf("NewAudioProcessor() error = %v", err)
	}

	return processor, bus, worker
}
func validAudioFrame() neurocall.AudioFrame {
	return neurocall.AudioFrame{
		Data:       []int16{1000, 1000, 1000},
		Timestamp:  0,
		SampleRate: 8000,
		Channels:   1,
	}
}
func (s *audioProcessorTestSegmenter) Process(
	ctx context.Context,
	frame neurocall.AudioFrame,
) ([]neurocall.AudioSegment, error) {
	s.processCalls++

	if s.processErr != nil {
		return nil, s.processErr
	}

	return append(
		[]neurocall.AudioSegment(nil),
		s.segments...,
	), nil
}
func (s *audioProcessorTestSegmenter) Flush(
	ctx context.Context,
) ([]neurocall.AudioSegment, error) {
	s.flushCalls++

	if s.flushErr != nil {
		return nil, s.flushErr
	}

	return append(
		[]neurocall.AudioSegment(nil),
		s.segments...,
	), nil
}
func (s *audioProcessorTestSegmenter) Reset() {
	s.resetCalls++
}
func (v *audioProcessorTestVAD) Process(
	ctx context.Context,
	frame neurocall.AudioFrame,
) (bool, error) {
	v.calls++

	if v.err != nil {
		return false, v.err
	}

	return v.speech, nil
}
func (e pipelineTestEncoder) Encode(pcm []int16) []byte {
	return e.output
}
func (d pipelineTestDecoder) Decode(data []byte) []int16 {
	return d.output
}
func (r *pipelineTestResampler) Resample(
	input []int16,
	fromRate int,
	toRate int,
) []int16 {
	r.called = true
	r.fromRate = fromRate
	r.toRate = toRate

	return r.output
}
func buildIntegrationINVITE(
	callID string,
	clientPort int,
	body []byte,
) []byte {

	return []byte(
		"INVITE sip:neurocall@127.0.0.1 SIP/2.0\r\n" +
			"Via: SIP/2.0/UDP 127.0.0.1:" +
			strconv.Itoa(clientPort) +
			";branch=z9hG4bK-test\r\n" +
			"From: <sip:test@127.0.0.1>;tag=test\r\n" +
			"To: <sip:neurocall@127.0.0.1>\r\n" +
			"Call-ID: " + callID + "\r\n" +
			"CSeq: 1 INVITE\r\n" +
			"Contact: <sip:test@127.0.0.1>\r\n" +
			"Content-Type: application/sdp\r\n" +
			"Content-Length: " +
			strconv.Itoa(len(body)) +
			"\r\n\r\n" +
			string(body),
	)
}
func (s *integrationSTT) Transcribe(
	ctx context.Context,
	segment neurocall.AudioSegment,
) (neurocall.Transcript, error) {

	select {
	case s.called <- segment:
	default:
	}

	return neurocall.Transcript{
		Text: "integration speech detected",
	}, nil
}
func (s *testSTT) Transcribe(
	ctx context.Context,
	segment neurocall.AudioSegment,
) (neurocall.Transcript, error) {
	return s.transcribe(ctx, segment)
}
func (l *integrationTestLLM) Chat(
	messages []neurocall.Message,
) (neurocall.Message, error) {
	l.calls++

	return l.response, nil
}
func (t *integrationTestTTS) Synthesize(
	ctx context.Context,
	text string,
) (neurocall.AudioStreamData, error) {
	t.calls++
	t.text = text

	return t.data, nil
}
func (t *testTTSFail) Synthesize(
	ctx context.Context,
	text string,
) (neurocall.AudioStreamData, error) {
	t.calls++

	if t.err != nil {
		return neurocall.AudioStreamData{}, t.err
	}

	return t.data, nil
}
func (t *testTTS) Synthesize(
	ctx context.Context,
	text string,
) (neurocall.AudioStreamData, error) {
	t.calls++
	t.texts = append(t.texts, text)
	if t.err != nil {
		return neurocall.AudioStreamData{}, t.err
	}
	return t.audio, nil
}
func (r *testClosableResource) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.closed = true
	return nil
}

func (r *testClosableResource) Closed() bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.closed
}
func (v *testVAD) Process(
	ctx context.Context,
	frame neurocall.AudioFrame,
) (bool, error) {

	v.mu.Lock()
	v.calls++
	result := v.result
	err := v.err
	v.mu.Unlock()

	return result, err
}
func (s *testSegmenter) Process(
	ctx context.Context,
	frame neurocall.AudioFrame,
) ([]neurocall.AudioSegment, error) {

	s.mu.Lock()
	defer s.mu.Unlock()

	s.processCalls++

	if s.err != nil {
		return nil, s.err
	}

	return append(
		[]neurocall.AudioSegment(nil),
		s.segments...,
	), nil
}
func (s *testSegmenter) Flush(
	ctx context.Context,
) ([]neurocall.AudioSegment, error) {

	s.mu.Lock()
	defer s.mu.Unlock()

	s.flushCalls++

	if s.err != nil {
		return nil, s.err
	}

	return append(
		[]neurocall.AudioSegment(nil),
		s.segments...,
	), nil
}
func (s *testSegmenter) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.resetCalls++
}
func (e *testSTTEngine) Transcribe(
	ctx context.Context,
	segment neurocall.AudioSegment,
) (string, error) {
	return "hello", nil
}
func newTestSTTWorker(
	bus *EventBus,
) (*STTWorker, error) {
	engine := NewSTTEngine(NewFakeSTT())
	return NewSTTWorker(
		engine,
		bus,
		"test-call",
		8,
	)
}
func (l *testLLM) Chat(
	messages []neurocall.Message,
) (neurocall.Message, error) {

	l.calls++

	copied := append(
		[]neurocall.Message(nil),
		messages...,
	)

	l.messages = append(
		l.messages,
		copied,
	)

	if l.err != nil {
		return neurocall.Message{}, l.err
	}

	return l.response, nil
}
func generate440HzWAV(duration time.Duration) []byte {
	sampleRate := 16000
	samples := int(duration.Seconds() * float64(sampleRate))

	pcm := make([]int16, samples)

	for i := 0; i < samples; i++ {
		t := float64(i) / float64(sampleRate)

		// 440Hz sine wave
		pcm[i] = int16(
			3000 * math.Sin(2*math.Pi*440*t),
		)
	}

	return openaipkg.PCM16ToWAV(
		pcm,
		sampleRate,
		1,
	)
}
func generate440Hz(
	sampleRate int,
	duration time.Duration,
) []int16 {

	totalSamples := int(
		float64(sampleRate) *
			duration.Seconds(),
	)

	pcm := make([]int16, totalSamples)

	frequency := 440.0

	amplitude := 16000.0

	for i := 0; i < totalSamples; i++ {

		t := float64(i) / float64(sampleRate)

		value := math.Sin(
			2 * math.Pi * frequency * t,
		)

		pcm[i] = int16(
			value * amplitude,
		)
	}

	return pcm
}
