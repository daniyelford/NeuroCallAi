package core

import (
	"context"
	"sync"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
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
