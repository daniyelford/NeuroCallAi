package core

import (
	"context"
	"encoding/binary"
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

// CodecRegistry
func NewCodecRegistry() *CodecRegistry {
	return &CodecRegistry{preferred: defaultCodecRegistry}
}
func (r *CodecRegistry) Negotiate(remote []neurocall.Codec) (neurocall.Codec, error) {
	for _, local := range r.preferred {
		for _, remoteCodec := range remote {
			if local.Name == remoteCodec.Name {
				return local, nil
			}
		}
	}
	return neurocall.Codec{}, neurocall.ErrNoCommonCodec
}

// CodecRegistry
// PortAllocator
func NewPortAllocator(min, max int) *PortAllocator {
	if min%2 != 0 {
		min++
	}
	available := make(map[int]struct{})
	for port := min; port <= max; port += 2 {
		available[port] = struct{}{}
	}
	return &PortAllocator{
		min:       min,
		max:       max,
		available: available,
	}
}
func (a *PortAllocator) Allocate() (int, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for port := a.min; port <= a.max; port += 2 {
		if _, ok := a.available[port]; ok {
			delete(a.available, port)
			return port, nil
		}
	}
	return 0, errRTPPortExhausted
}
func (a *PortAllocator) Release(port int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if port < a.min || port > a.max || port%2 != 0 {
		return
	}
	a.available[port] = struct{}{}
}

// PortAllocator
// AudioBuffer
func NewAudioBuffer(capacity int) *AudioBuffer {
	if capacity < 0 {
		capacity = 0
	}
	return &AudioBuffer{
		data:     make([]int16, 0, capacity),
		capacity: capacity,
	}
}
func (b *AudioBuffer) Write(data []int16) {
	if len(data) == 0 {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.data = append(b.data, data...)
	if b.capacity > 0 &&
		len(b.data) > b.capacity {
		excess := len(b.data) - b.capacity
		b.data = b.data[excess:]
	}
}
func (b *AudioBuffer) Read(n int) []int16 {
	b.mu.Lock()
	defer b.mu.Unlock()
	if n <= 0 || len(b.data) == 0 {
		return nil
	}
	if n > len(b.data) {
		n = len(b.data)
	}
	out := make([]int16, n)
	copy(out, b.data[:n])
	b.data = b.data[n:]
	return out
}
func (b *AudioBuffer) Len() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.data)
}
func (b *AudioBuffer) Clear() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.data = b.data[:0]
}

// AudioBuffer
// AudioPipeline
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
	select {
	case audio := <-p.output:
		return audio, nil
	case <-p.ctx.Done():
		return nil, neurocall.ErrCallClosed
	}
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

// AudioPipeline
// EventBus
func NewEventBus() *EventBus {
	return &EventBus{handlers: make(map[string][]EventHandler)}
}
func (b *EventBus) Subscribe(name string, handler EventHandler) {
	if handler == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.handlers[name] = append(b.handlers[name], handler)
}
func (b *EventBus) Publish(event Event) {
	b.mu.RLock()
	handlers := append([]EventHandler(nil), b.handlers[event.Name]...)
	b.mu.RUnlock()
	for _, handler := range handlers {
		handler(event)
	}
}

// EventBus
// ShutdownManager
func NewShutdownManager() *ShutdownManager {
	return &ShutdownManager{}
}
func (s *ShutdownManager) Register(
	handler func() error,
) {
	if handler == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.handlers =
		append(s.handlers, handler)
}
func (s *ShutdownManager) Shutdown() error {
	s.mu.Lock()

	if s.shuttingDown {
		s.mu.Unlock()
		return nil
	}

	s.shuttingDown = true

	handlers := append(
		[]func() error(nil),
		s.handlers...,
	)

	s.mu.Unlock()

	var firstErr error

	for _, handler := range handlers {
		if err := handler(); err != nil &&
			firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}

// ShutdownManager
// Container
func NewContainer() *Container {
	return &Container{
		services: make(map[string]any),
	}
}
func (c *Container) Set(
	name string,
	service any,
) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.services[name] = service
}
func (c *Container) Get(
	name string,
) (any, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	value, ok := c.services[name]

	return value, ok
}

// Container
// PluginRegistry
func NewPluginRegistry() *PluginRegistry {
	return &PluginRegistry{
		plugins: make(map[string]Plugin),
	}
}
func (r *PluginRegistry) Register(
	plugin Plugin,
) error {
	if plugin == nil {
		return fmt.Errorf(
			"cannot register nil plugin",
		)
	}

	name := plugin.Name()

	if name == "" {
		return fmt.Errorf(
			"plugin name is empty",
		)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.plugins[name]; exists {
		return fmt.Errorf(
			"plugin already registered: %s",
			name,
		)
	}

	r.plugins[name] = plugin

	return nil
}
func (r *PluginRegistry) Get(
	name string,
) (Plugin, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	plugin, ok := r.plugins[name]

	return plugin, ok
}
func (r *PluginRegistry) List() []Plugin {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(
		[]Plugin,
		0,
		len(r.plugins),
	)

	for _, plugin := range r.plugins {
		result = append(result, plugin)
	}

	return result
}

// PluginRegistry
// PluginGraph
func NewPluginGraph() *PluginGraph {
	return &PluginGraph{
		dependencies: make(
			map[string][]string,
		),
	}
}
func (g *PluginGraph) AddDependency(
	plugin string,
	dependency string,
) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.dependencies[plugin] =
		append(
			g.dependencies[plugin],
			dependency,
		)
}
func (g *PluginGraph) Dependencies(
	plugin string,
) []string {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return append(
		[]string(nil),
		g.dependencies[plugin]...,
	)
}

// PluginGraph
// CallManager
func NewCallManager(
	minRTPPort int,
	maxRTPPort int,
) *CallManager {

	return &CallManager{
		calls: make(map[string]*SIPCall),

		ports: NewPortAllocator(
			minRTPPort,
			maxRTPPort,
		),

		sessions: make(
			map[string]*CallSession,
		),
	}
}
func (m *CallManager) StartSession(
	call *SIPCall,
) (*CallSession, error) {

	session, err := m.CreateSession(call)

	if err != nil {
		return nil, err
	}

	if err := session.Start(); err != nil {
		return nil, err
	}

	return session, nil
}
func (m *CallManager) Add(
	call *SIPCall,
) error {

	if call == nil {
		return neurocall.ErrCallNotFound
	}

	if call.CallID == "" {
		return neurocall.ErrCallNotFound
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.calls[call.CallID]; exists {
		return fmt.Errorf(
			"call already exists: %s",
			call.CallID,
		)
	}

	m.calls[call.CallID] = call

	return nil
}
func (m *CallManager) Get(
	id string,
) (*SIPCall, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	call, ok := m.calls[id]
	if !ok {
		return nil, neurocall.ErrCallNotFound
	}
	return call, nil
}
func (m *CallManager) Remove(id string) error {
	if m == nil {
		return neurocall.ErrCallNotFound
	}

	m.mu.Lock()

	call, exists := m.calls[id]
	if !exists {
		m.mu.Unlock()
		return neurocall.ErrCallNotFound
	}

	session := m.sessions[id]

	delete(m.calls, id)
	delete(m.sessions, id)

	m.mu.Unlock()

	if call != nil {
		call.mu.Lock()
		localPort := call.LocalRTPPort
		call.LocalRTPPort = 0
		call.mu.Unlock()

		if localPort > 0 {
			m.ports.Release(localPort)
		}
	}

	if session != nil {
		_ = session.Close()
	}

	if call != nil {
		_ = call.CloseResources()
	}

	return nil
}
func (m *CallManager) List() []*SIPCall {

	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(
		[]*SIPCall,
		0,
		len(m.calls),
	)

	for _, call := range m.calls {
		result = append(
			result,
			call,
		)
	}

	return result
}
func (m *CallManager) CreateSession(
	call *SIPCall,
) (*CallSession, error) {
	if call == nil || call.CallID == "" {
		return nil, neurocall.ErrCallNotFound
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.calls[call.CallID]; !exists {
		return nil, neurocall.ErrCallNotFound
	}
	if _, exists := m.sessions[call.CallID]; exists {
		return nil, fmt.Errorf("session already exists: %s", call.CallID)
	}
	session, err := NewCallSession(call)
	if err != nil {
		return nil, err
	}
	// 	m.calls[call.CallID] = call
	m.sessions[call.CallID] = session
	return session, nil
}
func (m *CallManager) GetSession(
	id string,
) (*CallSession, error) {

	m.mu.RLock()
	defer m.mu.RUnlock()

	session, ok := m.sessions[id]

	if !ok {
		return nil, neurocall.ErrCallNotFound
	}

	return session, nil
}
func (m *CallManager) RemoveSession(id string) error {
	m.mu.Lock()
	session, ok := m.sessions[id]
	if !ok {
		m.mu.Unlock()
		return neurocall.ErrCallNotFound
	}
	delete(m.sessions, id)
	m.mu.Unlock()
	return session.Close()
}

// CallManager
// CallSession
func NewCallSession(call *SIPCall) (*CallSession, error) {
	if call == nil {
		return nil, neurocall.ErrCallNotFound
	}

	if call.CallID == "" {
		return nil, neurocall.ErrCallNotFound
	}

	ctx, cancel := context.WithCancel(
		context.Background(),
	)

	events := NewEventBus()

	conversation := NewConversation()

	session := &CallSession{
		ID:           call.CallID,
		Call:         call,
		Events:       events,
		Conversation: conversation,

		Memory: NewMemory(),
		Tools:  NewToolRegistry(),

		ctx:    ctx,
		cancel: cancel,
	}

	return session, nil
}
func (s *CallSession) ConfigureConversation(
	llm *LLMEngine,
) error {
	if s == nil {
		return neurocall.ErrCallClosed
	}

	if llm == nil {
		return fmt.Errorf("LLM engine is nil")
	}

	s.mu.Lock()

	if s.closed {
		s.mu.Unlock()
		return neurocall.ErrCallClosed
	}

	s.LLM = llm

	s.ConversationEngine =
		NewConversationEngine(
			llm,
			s.Events,
		)

	s.mu.Unlock()

	return s.AttachSTTEvents()
}
func (s *CallSession) ConfigureVoice(
	tts *TTSEngine,
) error {
	if s == nil {
		return neurocall.ErrCallClosed
	}

	if tts == nil {
		return fmt.Errorf("TTS engine is nil")
	}

	s.mu.Lock()

	if s.closed {
		s.mu.Unlock()
		return neurocall.ErrCallClosed
	}

	s.TTS = tts

	voice := NewVoiceResponseEngine(
		tts,
		s.Events,
	)

	s.VoiceEngine = voice

	call := s.Call

	s.mu.Unlock()

	if call != nil {
		voice.RegisterCall(call)
	}

	return nil
}
func (s *CallSession) ConfigureSTT(
	worker *STTWorker,
) error {
	if s == nil {
		return neurocall.ErrCallClosed
	}

	if worker == nil {
		return fmt.Errorf("STT worker is nil")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return neurocall.ErrCallClosed
	}

	s.STTWorker = worker

	return nil
}
func (s *CallSession) AttachSTTEvents() error {
	if s == nil {
		return neurocall.ErrCallClosed
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return neurocall.ErrCallClosed
	}
	if s.sttEventsAttached {
		s.mu.Unlock()
		return nil
	}
	bus := s.Events
	engine := s.ConversationEngine
	if bus == nil {
		s.mu.Unlock()
		return fmt.Errorf("event bus is nil")
	}
	if engine == nil {
		s.mu.Unlock()
		return fmt.Errorf(
			"conversation engine is not configured",
		)
	}
	s.sttEventsAttached = true
	s.mu.Unlock()
	bus.Subscribe(
		EventTranscript,
		func(event Event) {
			data, ok := event.Data.(TranscriptEvent)
			if !ok {
				return
			}
			if data.CallID != s.ID {
				return
			}
			go func() {
				if err := engine.HandleTranscript(data); err != nil {
					bus.Publish(Event{Name: EventLLMError, Data: err})
				}
			}()
		},
	)
	return nil
}
func (s *CallSession) ProcessAudioSegment(
	segment neurocall.AudioSegment,
) error {

	if s == nil {
		return neurocall.ErrCallClosed
	}

	if len(segment.Data) == 0 {
		return neurocall.ErrInvalidAudio
	}

	s.mu.RLock()

	if s.closed || !s.running {
		s.mu.RUnlock()
		return neurocall.ErrCallClosed
	}

	worker := s.STTWorker

	s.mu.RUnlock()

	if worker == nil {
		return fmt.Errorf(
			"STT worker is not configured",
		)
	}

	return worker.Push(segment)
}
func (s *CallSession) ProcessAudioFrame(
	ctx context.Context,
	frame neurocall.AudioFrame,
	pcm []int16,
) error {
	if s == nil {
		return neurocall.ErrCallClosed
	}
	if len(pcm) == 0 {
		return neurocall.ErrInvalidAudio
	}
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	s.mu.RLock()
	if s.closed || !s.running {
		s.mu.RUnlock()
		return neurocall.ErrCallClosed
	}
	call := s.Call
	worker := s.STTWorker
	s.mu.RUnlock()
	if call == nil {
		return neurocall.ErrCallNotFound
	}
	if worker == nil {
		return fmt.Errorf("STT worker is not configured")
	}
	call.mu.RLock()
	segmenter := call.Segmenter
	call.mu.RUnlock()
	if segmenter == nil {
		return fmt.Errorf("speech segmenter is not configured")
	}
	segments, err := segmenter.Process(ctx, frame)
	if err != nil {
		return err
	}
	for _, segment := range segments {
		if len(segment.Data) == 0 {
			continue
		}
		if err := worker.Push(segment); err != nil {
			return err
		}
	}
	return nil
}
func (s *CallSession) SetSTT(
	stt neurocall.STT,
) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.STT = stt
}
func (s *CallSession) SetStreamingSTT(
	stt neurocall.StreamingSTT,
) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.StreamingSTT = stt
}
func (s *CallSession) SetTTS(
	tts neurocall.TTS,
) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.TTS = tts
}
func (s *CallSession) SetLLM(
	llm neurocall.LLM,
) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.LLM = llm
}
func (s *CallSession) SetMemory(
	memory neurocall.Memory,
) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Memory = memory
}
func (s *CallSession) SetTools(
	tools neurocall.ToolRegistry,
) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Tools = tools
}
func (s *CallSession) Context() context.Context {

	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.ctx
}
func (s *CallSession) Start() error {
	if s == nil {
		return neurocall.ErrCallClosed
	}

	s.mu.Lock()

	if s.closed {
		s.mu.Unlock()
		return neurocall.ErrCallClosed
	}

	if s.running {
		s.mu.Unlock()
		return nil
	}

	s.running = true

	audio := s.Audio
	worker := s.STTWorker

	s.mu.Unlock()

	if audio != nil {
		audio.Start()
	}

	if worker != nil {
		if err := worker.Start(); err != nil {
			if audio != nil {
				_ = audio.Close()
			}

			s.mu.Lock()
			s.running = false
			s.mu.Unlock()

			return err
		}
	}

	// if err := s.AttachSTTEventsIfReady(); err != nil {
	// 	s.mu.Lock()
	// 	s.running = false
	// 	s.mu.Unlock()

	// 	return err
	// }

	if s.Events != nil {
		s.Events.Publish(Event{
			Name: EventCallStarted,
			Data: s,
		})
	}

	return nil
}
func (s *CallSession) State() CallSessionState {
	if s == nil {
		return CallSessionState{
			Closed: true,
		}
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return CallSessionState{
		ID: s.ID,

		Running: s.running,
		Closed:  s.closed,

		HasAudio: s.Audio != nil,
		HasSTT:   s.STTWorker != nil,
		HasLLM:   s.ConversationEngine != nil,
		HasTTS:   s.VoiceEngine != nil,
	}
}
func (s *CallSession) PushAudio(
	audio []int16,
) error {

	if s == nil {
		return neurocall.ErrCallClosed
	}

	if len(audio) == 0 {
		return neurocall.ErrInvalidAudio
	}

	s.mu.RLock()

	if s.closed || !s.running {
		s.mu.RUnlock()
		return neurocall.ErrCallClosed
	}

	pipeline := s.Audio

	s.mu.RUnlock()

	if pipeline == nil {
		return fmt.Errorf("audio pipeline is not configured")
	}

	return pipeline.Push(audio)
}
func (s *CallSession) PushSTTSegment(
	segment neurocall.AudioSegment,
) error {

	if s == nil {
		return neurocall.ErrCallClosed
	}

	if len(segment.Data) == 0 {
		return neurocall.ErrInvalidAudio
	}

	s.mu.RLock()

	if s.closed || !s.running {
		s.mu.RUnlock()
		return neurocall.ErrCallClosed
	}

	worker := s.STTWorker

	s.mu.RUnlock()

	if worker == nil {
		return fmt.Errorf("STT worker is not configured")
	}

	return worker.Push(segment)
}
func (s *CallSession) Running() bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.running && !s.closed
}
func (s *CallSession) Closed() bool {
	if s == nil {
		return true
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.closed
}
func (s *CallSession) SetSTTWorker(worker *STTWorker) error {
	// if worker == nil {
	// 	return fmt.Errorf("STT worker is nil")
	// }
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return neurocall.ErrCallClosed
	}
	s.STTWorker = worker
	return nil
}
func (s *CallSession) GetSTTWorker() *STTWorker {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.STTWorker
}
func (s *CallSession) Close() error {
	if s == nil {
		return nil
	}

	s.mu.Lock()

	if s.closed {
		s.mu.Unlock()
		return nil
	}

	s.closed = true
	s.running = false

	cancel := s.cancel
	worker := s.STTWorker
	audio := s.Audio
	events := s.Events

	s.mu.Unlock()

	// Stop STT first.
	if worker != nil {
		worker.Stop()
	}

	// Stop audio pipeline.
	if audio != nil {
		_ = audio.Close()
	}

	// Cancel session context.
	if cancel != nil {
		cancel()
	}

	// Notify observers.
	if events != nil {
		events.Publish(Event{
			Name: EventCallEnded,
			Data: s,
		})
	}

	return nil
}

// CallSession
// MapMemory
func NewMemory() *MapMemory {
	return &MapMemory{
		data: make(map[string]any),
	}
}
func (m *MapMemory) Get(
	key string,
) (any, bool) {

	m.mu.RLock()
	defer m.mu.RUnlock()

	value, ok := m.data[key]

	return value, ok
}

func (m *MapMemory) Set(
	key string,
	value any,
) {

	if key == "" {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.data[key] = value
}

func (m *MapMemory) Delete(
	key string,
) {

	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.data, key)
}
func (m *MapMemory) Clear() {

	m.mu.Lock()
	defer m.mu.Unlock()

	m.data = make(map[string]any)
}

// MapMemory
// ToolRegistry
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		tools: make(map[string]neurocall.Tool),
	}
}
func (r *ToolRegistry) Register(
	tool neurocall.Tool,
) error {
	if tool == nil {
		return neurocall.ErrToolNotFound
	}
	name := tool.Name()
	if name == "" {
		return fmt.Errorf(
			"tool name is empty",
		)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[name]; exists {
		return fmt.Errorf(
			"tool already registered: %s",
			name,
		)
	}
	r.tools[name] = tool
	return nil
}
func (r *ToolRegistry) Get(
	name string,
) (neurocall.Tool, bool) {

	r.mu.RLock()
	defer r.mu.RUnlock()

	tool, ok := r.tools[name]

	return tool, ok
}
func (r *ToolRegistry) List() []neurocall.Tool {

	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(
		[]neurocall.Tool,
		0,
		len(r.tools),
	)

	for _, tool := range r.tools {
		result = append(result, tool)
	}

	return result
}
func (r *ToolRegistry) Call(
	name string,
	args map[string]any,
) (any, error) {

	tool, ok := r.Get(name)

	if !ok {
		return nil, neurocall.ErrToolNotFound
	}

	return tool.Call(args)
}

// ToolRegistry
// SIPCall
func (c *SIPCall) ID() string {
	return c.CallID
}
func (c *SIPCall) Answer() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return neurocall.ErrCallClosed
	}
	if c.answered {
		return nil
	}
	c.answered = true
	return nil
}
func (c *SIPCall) Hangup() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	rtp := c.RTP
	pipeline := c.Pipeline
	stt := c.STT
	c.mu.Unlock()
	if stt != nil {
		stt.Stop()
	}
	if rtp != nil {
		_ = rtp.Close()
	}
	if pipeline != nil {
		_ = pipeline.Close()
	}
	return nil
}
func (c *SIPCall) Answered() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.answered && !c.closed
}
func (c *SIPCall) Closed() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.closed
}
func (c *SIPCall) SetSTTWorker(
	worker *STTWorker,
) error {

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return neurocall.ErrCallClosed
	}

	c.STT = worker

	return nil
}
func (c *SIPCall) STTWorker() *STTWorker {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.STT
}
func (c *SIPCall) StartReceiveLoop() {
	if c == nil {
		return
	}
	c.mu.Lock()
	if c.closed || c.receiveRunning {
		c.mu.Unlock()
		return
	}
	c.receiveRunning = true
	c.mu.Unlock()
	go func() {
		defer func() {
			c.mu.Lock()
			c.receiveRunning = false
			c.mu.Unlock()
		}()
		for {
			c.mu.RLock()
			if c.closed {
				c.mu.RUnlock()
				return
			}
			rtp := c.RTP
			pipeline := c.Pipeline
			segmenter := c.Segmenter
			stt := c.STT
			codec := c.Codec
			c.mu.RUnlock()
			if rtp == nil ||
				pipeline == nil ||
				segmenter == nil {
				return
			}
			decoder := pipeline.Decoder()
			if decoder == nil {
				return
			}
			pcm, err := rtp.ReadPCM(decoder)
			if err != nil {
				if c.Closed() {
					return
				}
				continue
			}
			if len(pcm) == 0 {
				continue
			}
			frame := neurocall.AudioFrame{
				Data:       pcm,
				SampleRate: codec.ClockRate,
				Channels:   codec.Channels,
			}
			ctx := context.Background()
			segments, err := segmenter.Process(ctx, frame)
			if err != nil {
				continue
			}
			if stt == nil {
				continue
			}
			for _, segment := range segments {
				if len(segment.Data) == 0 {
					continue
				}
				if err := stt.Push(segment); err != nil {
					if c.Closed() {
						return
					}

					continue
				}
			}
		}
	}()
}
func (c *SIPCall) StartRTP(
	localIP string,
) error {

	c.mu.Lock()

	if c.closed {
		c.mu.Unlock()
		return neurocall.ErrCallClosed
	}

	if c.RTP != nil {
		c.mu.Unlock()
		return nil
	}

	codec := c.Codec
	remoteIP := c.RemoteIP
	remotePort := c.RemotePort
	localPort := c.LocalRTPPort

	c.mu.Unlock()

	rtp, err := NewRTPSession(
		localIP,
		localPort,
		remoteIP,
		remotePort,
		codec,
	)

	if err != nil {
		return err
	}

	c.mu.Lock()

	if c.closed {
		c.mu.Unlock()
		_ = rtp.Close()
		return neurocall.ErrCallClosed
	}

	c.RTP = rtp

	c.mu.Unlock()

	return nil
}
func (c *SIPCall) StartAudio(
	cfg AudioConfig,
) error {

	c.mu.Lock()

	if c.closed {
		c.mu.Unlock()
		return neurocall.ErrCallClosed
	}

	if c.Pipeline != nil {
		c.mu.Unlock()
		return nil
	}

	c.mu.Unlock()

	pipeline := NewAudioPipeline(cfg)

	encoder, err :=
		EncoderFor(c.Codec)

	if err != nil {
		return err
	}

	decoder, err :=
		DecoderFor(c.Codec)

	if err != nil {
		return err
	}

	pipeline.SetEncoder(encoder)
	pipeline.SetDecoder(decoder)

	pipeline.Start()

	c.mu.Lock()

	if c.closed {
		c.mu.Unlock()
		_ = pipeline.Close()
		return neurocall.ErrCallClosed
	}

	c.Pipeline = pipeline

	c.mu.Unlock()

	return nil
}
func (c *SIPCall) Speak(
	ctx context.Context,
	text string,
) error {

	c.mu.RLock()

	tts := c.TTS
	rtp := c.RTP
	pipeline := c.Pipeline
	codec := c.Codec
	closed := c.closed

	c.mu.RUnlock()

	if closed {
		return neurocall.ErrCallClosed
	}

	if tts == nil {
		return fmt.Errorf(
			"TTS is not configured",
		)
	}

	if rtp == nil {
		return fmt.Errorf(
			"RTP is not configured",
		)
	}

	if pipeline == nil {
		return fmt.Errorf(
			"audio pipeline is not configured",
		)
	}

	audio, err := tts.Synthesize(
		ctx,
		text,
	)
	if err != nil {
		return err
	}
	pcm := BytesToPCM16(audio.Data)
	if len(pcm) == 0 {
		return neurocall.ErrInvalidAudio
	}
	pcm = pipeline.Resample(
		pcm,
		audio.Format.SampleRate,
		codec.ClockRate,
	)
	encoder := pipeline.Encoder()
	if encoder == nil {
		return fmt.Errorf(
			"audio encoder is not configured",
		)
	}
	return rtp.WritePCMRealtime(
		ctx,
		encoder,
		pcm,
	)
}
func (c *SIPCall) CloseResources() error {
	c.mu.Lock()
	rtp := c.RTP
	pipeline := c.Pipeline
	tts := c.TTS
	c.RTP = nil
	c.Pipeline = nil
	c.TTS = nil
	c.closed = true
	c.mu.Unlock()
	var firstErr error
	if rtp != nil {
		if err := rtp.Close(); err != nil {
			firstErr = err
		}
	}
	if pipeline != nil {
		if err := pipeline.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if tts != nil {
		if err := tts.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// SIPCall
// SIPServer
func NewSIPServer(
	config SIPConfig,
	manager *CallManager,
	codecs *CodecRegistry,
	stt neurocall.STT,
	llm neurocall.LLM,
	tts neurocall.TTS,
) *SIPServer {

	if manager == nil {
		manager = NewCallManager(
			config.RTPMinPort,
			config.RTPMaxPort,
		)
	}

	if codecs == nil {
		codecs = NewCodecRegistry()
	}
	bus := NewEventBus()
	server := &SIPServer{
		manager: manager,
		codecs:  codecs,
		config:  config,
		bus:     bus,
	}

	if stt != nil {
		server.stt = NewSTTEngine(stt)
	}

	if llm != nil {
		server.llm = NewLLMEngine(llm)
	}

	if tts != nil {
		server.tts = NewTTSEngine(tts)
	}
	if server.llm != nil {
		server.conversation =
			NewConversationEngine(
				server.llm,
				bus,
			)
	}

	if server.tts != nil {
		server.voice =
			NewVoiceResponseEngine(
				server.tts,
				bus,
			)
	}

	return server
}
func (s *SIPServer) Listen(
	ctx context.Context,
) error {

	s.mu.Lock()

	if s.conn != nil {
		s.mu.Unlock()

		return fmt.Errorf(
			"SIP server already listening",
		)
	}

	addr := &net.UDPAddr{
		IP: net.ParseIP(
			s.config.ListenIP,
		),
		Port: s.config.SIPPort,
	}

	conn, err := net.ListenUDP(
		"udp",
		addr,
	)

	if err != nil {
		s.mu.Unlock()
		return err
	}

	serverCtx, cancel :=
		context.WithCancel(ctx)

	s.conn = conn
	s.ctx = serverCtx
	s.cancel = cancel

	s.mu.Unlock()

	go s.readLoop(serverCtx)

	return nil
}
func (s *SIPServer) Stop() error {

	s.mu.Lock()

	conn := s.conn
	cancel := s.cancel

	s.conn = nil
	s.cancel = nil

	s.mu.Unlock()

	if cancel != nil {
		cancel()
	}

	if conn != nil {
		return conn.Close()
	}

	return nil
}
func (s *SIPServer) readLoop(
	ctx context.Context,
) {

	buffer := make(
		[]byte,
		65535,
	)

	for {

		select {
		case <-ctx.Done():
			return

		default:
		}

		s.mu.RLock()
		conn := s.conn
		s.mu.RUnlock()

		if conn == nil {
			return
		}

		n, remote, err :=
			conn.ReadFromUDP(buffer)

		if err != nil {
			select {
			case <-ctx.Done():
				return
			default:
				continue
			}
		}

		data := append(
			[]byte(nil),
			buffer[:n]...,
		)

		go func(
			data []byte,
			remote *net.UDPAddr,
		) {

			_ = s.handle(
				ctx,
				data,
				remote,
			)

		}(data, remote)
	}
}
func (s *SIPServer) handle(
	ctx context.Context,
	data []byte,
	remote *net.UDPAddr,
) error {

	message, err :=
		ParseSIPMessage(data)

	if err != nil {
		return err
	}

	switch message.Method() {

	case "INVITE":
		return s.handleINVITE(
			ctx,
			message,
			remote,
		)

	case "ACK":
		return s.handleACK(
			ctx,
			message,
			remote,
		)

	case "BYE":
		return s.handleBYE(
			ctx,
			message,
			remote,
		)

	case "OPTIONS":
		return s.handleOPTIONS(
			ctx,
			message,
			remote,
		)

	default:
		return s.sendResponse(
			message,
			remote,
			501,
			"Not Implemented",
			nil,
		)
	}
}
func (s *SIPServer) sendResponse(
	req *SIPMessage,
	remote *net.UDPAddr,
	code int,
	reason string,
	body []byte,
) error {

	data := BuildSIPResponse(
		req,
		code,
		reason,
		body,
	)

	s.mu.RLock()
	conn := s.conn
	s.mu.RUnlock()

	if conn == nil {
		return neurocall.ErrCallClosed
	}

	_, err := conn.WriteToUDP(
		data,
		remote,
	)

	return err
}
func (s *SIPServer) handleINVITE(_ context.Context, req *SIPMessage, remote *net.UDPAddr) error {
	if err := s.sendResponse(req, remote, 100, "Trying", nil); err != nil {
		return err
	}
	if s.stt == nil {
		return s.sendResponse(req, remote, 503, "Service Unavailable", nil)
	}
	media, err := ParseSDP(req.Body)
	if err != nil {
		return s.sendResponse(req, remote, 488, "Not Acceptable Here", nil)
	}
	codec, err := s.codecs.Negotiate(media.Codecs)
	if err != nil {
		return s.sendResponse(req, remote, 488, "Not Acceptable Here", nil)
	}
	callID := req.Headers["call-id"]
	if callID == "" {
		return s.sendResponse(req, remote, 400, "Bad Request", nil)
	}
	var ttsPlayback *TTSPlayback
	if s.tts != nil {
		ttsPlayback, err = NewTTSPlayback(s.tts)
		if err != nil {
			return err
		}
	}
	cleanup := func() {
		if s.voice != nil {
			_ = s.voice.RemoveCall(callID)
		}

		_ = s.manager.Remove(callID)
	}
	// --------------------
	// Create call
	// --------------------
	call := &SIPCall{
		CallID:     callID,
		RemoteIP:   media.IP,
		RemotePort: media.Port,
		RemoteAddr: remote,
		Codec:      codec,
		TTS:        ttsPlayback,
	}
	if s.voice != nil {
		if err := s.voice.AddCall(call); err != nil {
			cleanup()
			return err
		}
	}
	// --------------------
	// Register call
	// --------------------
	if err := s.manager.Add(call); err != nil {
		cleanup()
		return err
	}
	session, err := s.manager.CreateSession(call)
	if err != nil {
		cleanup()
		return err
	}

	sttWorker, err := NewSTTWorker(
		s.stt,
		session.Events,
		callID,
		16,
	)
	if err != nil {
		cleanup()
		return err
	}

	if err := session.SetSTTWorker(sttWorker); err != nil {
		cleanup()
		return err
	}

	if err := call.SetSTTWorker(sttWorker); err != nil {
		cleanup()
		return err
	}

	if s.llm != nil {
		if err := session.ConfigureConversation(s.llm); err != nil {
			cleanup()
			return err
		}
	}

	if s.tts != nil {
		if err := session.ConfigureVoice(s.tts); err != nil {
			cleanup()
			return err
		}
	}

	// Allocate RTP
	port, err := s.manager.ports.Allocate()
	if err != nil {
		cleanup()
		return err
	}

	call.mu.Lock()
	call.LocalRTPPort = port
	call.mu.Unlock()

	// Start RTP
	if err := call.StartRTP(s.config.ListenIP); err != nil {
		cleanup()
		return err
	}

	// Start Audio
	if err := call.StartAudio(s.configToAudioConfig()); err != nil {
		cleanup()
		return err
	}

	// VAD + Segmenter
	vad := NewEnergyVAD(500)

	segmenter := NewAudioSegmenter(
		vad,
		s.configToAudioConfig(),
	)

	call.mu.Lock()
	call.VAD = vad
	call.Segmenter = segmenter
	call.mu.Unlock()

	// Start Session
	if err := session.Start(); err != nil {
		cleanup()
		return err
	}

	// Start RTP receive → STT
	// call.StartReceiveLoop()

	// SDP Answer
	body := BuildSDPAnswer(call, s.config)
	// --------------------
	// 200 OK
	// --------------------
	if err := s.sendResponse(req, remote, 200, "OK", body); err != nil {
		cleanup()
		return err
	}
	return nil
}
func (s *SIPServer) handleACK(
	_ context.Context,
	req *SIPMessage,
	_ *net.UDPAddr,
) error {
	callID := req.Headers["call-id"]
	if callID == "" {
		return neurocall.ErrCallNotFound
	}
	call, err := s.manager.Get(callID)
	if err != nil {
		return err
	}
	if !call.Answered() {
		if err := call.Answer(); err != nil {
			return err
		}
		call.StartReceiveLoop()
	}
	session, err := s.manager.GetSession(callID)
	if err == nil {
		session.Events.Publish(Event{
			Name: EventCallAnswered,
			Data: session,
		})
	}
	return nil
}
func (s *SIPServer) handleBYE(
	_ context.Context,
	req *SIPMessage,
	remote *net.UDPAddr,
) error {
	callID := req.Headers["call-id"]
	if callID == "" {
		return s.sendResponse(req, remote, 400, "Bad Request", nil)
	}
	call, err := s.manager.Get(callID)
	if err != nil {
		return s.sendResponse(req, remote, 481, "Call/Transaction Does Not Exist", nil)
	}
	if err := s.sendResponse(req, remote, 200, "OK", nil); err != nil {
		return err
	}
	_ = call.Hangup()
	if s.voice != nil {
		_ = s.voice.RemoveCall(callID)
	}
	return s.manager.Remove(callID)
}
func (s *SIPServer) handleOPTIONS(
	_ context.Context,
	req *SIPMessage,
	remote *net.UDPAddr,
) error {

	return s.sendResponse(
		req,
		remote,
		200,
		"OK",
		nil,
	)
}
func (s *SIPServer) MediaIP() string {
	if s.config.ExternalIP != "" {
		return s.config.ExternalIP
	}
	if s.config.ListenIP != "" &&
		s.config.ListenIP != "0.0.0.0" &&
		s.config.ListenIP != "::" {
		return s.config.ListenIP
	}
	return "127.0.0.1"
}
func (s *SIPServer) configToAudioConfig() AudioConfig {
	return AudioConfig{
		SampleRate: DefaultAudioSampleRate,
		Channels:   DefaultAudioChannels,
		FrameSize:  DefaultAudioFrameSize,
		BufferSize: 32,
	}
}

// SIPServer
// SIPMessage
func (m *SIPMessage) Method() string {
	fields := strings.Fields(m.StartLine)

	if len(fields) == 0 {
		return ""
	}

	return fields[0]
}

// SIPMessage
// PCMU
func (PCMU) Encode(pcm []int16) []byte {
	out := make([]byte, len(pcm))

	for i, sample := range pcm {
		out[i] = linearToMuLaw(sample)
	}

	return out
}
func (PCMU) Decode(data []byte) []int16 {
	out := make([]int16, len(data))

	for i, sample := range data {
		out[i] = muLawToLinear(sample)
	}

	return out
}

// PCMU
// PCMA
func (PCMA) Encode(pcm []int16) []byte {
	out := make([]byte, len(pcm))

	for i, sample := range pcm {
		out[i] = linearToALaw(sample)
	}

	return out
}
func (PCMA) Decode(data []byte) []int16 {
	out := make([]int16, len(data))

	for i, sample := range data {
		out[i] = aLawToLinear(sample)
	}

	return out
}

// PCMA
// AudioEngine
func NewAudioEngine(
	pipeline *AudioPipeline,
) *AudioEngine {

	return &AudioEngine{
		pipeline: pipeline,
	}
}
func (e *AudioEngine) Start() error {

	e.mu.Lock()

	if e.running {
		e.mu.Unlock()
		return nil
	}

	e.running = true

	e.mu.Unlock()

	e.pipeline.Start()

	return nil
}

// AudioEngine
// STTEngine
func NewSTTEngine(
	stt neurocall.STT,
) *STTEngine {

	ctx, cancel := context.WithCancel(
		context.Background(),
	)

	return &STTEngine{
		stt:    stt,
		ctx:    ctx,
		cancel: cancel,
	}
}
func (e *STTEngine) SetSTT(
	stt neurocall.STT,
) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.stt = stt
}
func (e *STTEngine) Transcribe(
	ctx context.Context,
	segment neurocall.AudioSegment,
) (neurocall.Transcript, error) {

	e.mu.RLock()
	stt := e.stt
	engineCtx := e.ctx
	e.mu.RUnlock()

	if stt == nil {
		return neurocall.Transcript{}, fmt.Errorf(
			"STT is not configured",
		)
	}

	if ctx == nil {
		ctx = engineCtx
	}

	return stt.Transcribe(
		ctx,
		segment,
	)
}
func (e *STTEngine) Close() {

	e.mu.Lock()

	cancel := e.cancel
	e.cancel = nil

	e.mu.Unlock()

	if cancel != nil {
		cancel()
	}
}

// STTEngine
// STTWorker
func NewSTTWorker(
	engine *STTEngine,
	bus *EventBus,
	callID string,
	bufferSize int,
) (*STTWorker, error) {

	if engine == nil {
		return nil, fmt.Errorf("STT engine is nil")
	}

	if bus == nil {
		return nil, fmt.Errorf("event bus is nil")
	}

	if callID == "" {
		return nil, fmt.Errorf("call ID is empty")
	}

	if bufferSize <= 0 {
		bufferSize = 16
	}

	ctx, cancel := context.WithCancel(
		context.Background(),
	)

	return &STTWorker{
		engine: engine,
		bus:    bus,
		callID: callID,
		input:  make(chan neurocall.AudioSegment, bufferSize),
		ctx:    ctx,
		cancel: cancel,
	}, nil
}
func (w *STTWorker) SetStreamingSTT(
	stt neurocall.StreamingSTT,
) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.streaming = stt
}
func (w *STTWorker) Push(
	segment neurocall.AudioSegment,
) error {

	if len(segment.Data) == 0 {
		return neurocall.ErrInvalidAudio
	}

	w.mu.Lock()

	if w.stopped {
		w.mu.Unlock()
		return neurocall.ErrCallClosed
	}

	input := w.input
	ctx := w.ctx

	w.mu.Unlock()

	select {

	case <-ctx.Done():
		return neurocall.ErrCallClosed

	case input <- segment:
		return nil
	}
}
func (w *STTWorker) Start() error {
	w.mu.Lock()

	if w.stopped {
		w.mu.Unlock()
		return neurocall.ErrCallClosed
	}

	if w.running {
		w.mu.Unlock()
		return nil
	}

	w.running = true
	streaming := w.streaming
	ctx := w.ctx

	w.mu.Unlock()

	if streaming != nil {
		if err := streaming.Start(ctx); err != nil {
			w.mu.Lock()
			w.running = false
			w.mu.Unlock()
			return err
		}

		go w.runStreaming(streaming)
	}

	go w.run()

	return nil
}
func (w *STTWorker) runStreaming(streaming neurocall.StreamingSTT) {
	events := streaming.Events()

	for {
		select {
		case <-w.ctx.Done():
			return
		case event, ok := <-events:
			if !ok {
				return
			}
			w.bus.Publish(Event{
				Name: EventSTTEvent,
				Data: STTEventData{
					CallID: w.callID,
					Event:  event,
				},
			})
		}
	}
}
func (w *STTWorker) run() {
	for {
		select {

		case <-w.ctx.Done():
			return

		case segment, ok := <-w.input:
			if !ok {
				return
			}

			transcript, err := w.engine.Transcribe(
				w.ctx,
				segment,
			)

			if err != nil {
				w.bus.Publish(Event{
					Name: EventSTTError,
					Data: STTErrorEvent{
						CallID: w.callID,
						Err:    err,
					},
				})

				continue
			}

			w.bus.Publish(Event{
				Name: EventTranscript,
				Data: TranscriptEvent{
					CallID:     w.callID,
					Transcript: transcript,
				},
			})
		}
	}
}
func (w *STTWorker) Stop() {
	w.mu.Lock()

	if w.stopped {
		w.mu.Unlock()
		return
	}

	w.stopped = true
	w.running = false

	cancel := w.cancel
	streaming := w.streaming

	w.mu.Unlock()

	if cancel != nil {
		cancel()
	}

	if streaming != nil {
		_ = streaming.Close()
	}
}

// STTWorker
// EnergyVAD
func NewEnergyVAD(threshold int64) *EnergyVAD {
	if threshold <= 0 {
		threshold = 500
	}

	return &EnergyVAD{
		Threshold: threshold,
	}
}
func (v *EnergyVAD) Process(
	ctx context.Context,
	frame neurocall.AudioFrame,
) (bool, error) {

	select {
	case <-ctx.Done():
		return false, ctx.Err()

	default:
	}

	if len(frame.Data) == 0 {
		return false, nil
	}

	var energy int64

	for _, sample := range frame.Data {
		value := int64(sample)

		if value < 0 {
			value = -value
		}

		energy += value
	}

	energy /= int64(len(frame.Data))

	return energy >= v.Threshold, nil
}

// EnergyVAD
// AudioSegmenter
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

// AudioSegmenter
// LLMEngine
func NewLLMEngine(
	llm neurocall.LLM,
) *LLMEngine {
	return &LLMEngine{
		llm: llm,
	}
}
func (e *LLMEngine) SetLLM(
	llm neurocall.LLM,
) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.llm = llm
}
func (e *LLMEngine) Chat(
	messages []neurocall.Message,
) (neurocall.Message, error) {
	e.mu.RLock()
	llm := e.llm
	e.mu.RUnlock()
	if llm == nil {
		return neurocall.Message{},
			fmt.Errorf(
				"LLM is not configured",
			)
	}
	return llm.Chat(messages)
}

// LLMEngine
// CallMemory
func NewCallMemory() *CallMemory {
	return &CallMemory{
		values: make(map[string]any),
	}
}
func (m *CallMemory) Set(
	key string,
	value any,
) {
	if m == nil || key == "" {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.values[key] = value
}
func (m *CallMemory) Get(
	key string,
) (any, bool) {
	if m == nil || key == "" {
		return nil, false
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	value, ok := m.values[key]

	return value, ok
}
func (m *CallMemory) Delete(
	key string,
) {
	if m == nil || key == "" {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.values, key)
}
func (m *CallMemory) Clear() {
	if m == nil {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.values = make(map[string]any)
}

// CallMemory
// Conversation
func NewConversation() *Conversation {
	return &Conversation{
		messages: make([]neurocall.Message, 0),
		memory:   NewCallMemory(),
	}
}
func (c *Conversation) AddMessage(
	message neurocall.Message,
) {
	if c == nil {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.messages = append(
		c.messages,
		message,
	)
}
func (c *Conversation) Messages() []neurocall.Message {
	if c == nil {
		return nil
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	return append(
		[]neurocall.Message(nil),
		c.messages...,
	)
}
func (c *Conversation) Clear() {
	if c == nil {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.messages = c.messages[:0]
}
func (c *Conversation) Memory() *CallMemory {
	if c == nil {
		return nil
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.memory
}
func (c *Conversation) SetMemory(
	memory *CallMemory,
) {
	if c == nil || memory == nil {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.memory = memory
}

// func (c *Conversation) GenerateResponse(
//     fn func() error,
// ) error {
//     c.responseMu.Lock()
//     defer c.responseMu.Unlock()

//	    return fn()
//	}
//
// Conversation
// ConversationEngine
func NewConversationEngine(
	llm *LLMEngine,
	bus *EventBus,
) *ConversationEngine {
	return &ConversationEngine{
		llm:           llm,
		bus:           bus,
		conversations: make(map[string]*Conversation),
	}
}
func (e *ConversationEngine) GetOrCreate(
	callID string,
) (*Conversation, error) {

	if callID == "" {
		return nil, fmt.Errorf("call ID is empty")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if conversation, exists := e.conversations[callID]; exists {
		return conversation, nil
	}

	conversation := NewConversation()

	e.conversations[callID] = conversation

	return conversation, nil
}
func (e *ConversationEngine) HandleTranscript(
	event TranscriptEvent,
) error {

	if event.CallID == "" {
		return fmt.Errorf("call ID is empty")
	}

	text := strings.TrimSpace(
		event.Transcript.Text,
	)

	if text == "" {
		return nil
	}

	conversation, err := e.GetOrCreate(
		event.CallID,
	)

	if err != nil {
		return err
	}

	message := neurocall.Message{
		Role:    "user",
		Content: text,
	}

	conversation.AddMessage(message)

	return e.generateResponse(
		event.CallID,
		conversation,
	)
}
func (e *ConversationEngine) generateResponse(
	callID string,
	conversation *Conversation,
) error {

	e.mu.RLock()

	llm := e.llm
	bus := e.bus

	e.mu.RUnlock()

	if llm == nil {
		return fmt.Errorf("LLM is not configured")
	}

	messages := conversation.Messages()

	response, err := llm.Chat(messages)

	if err != nil {
		if bus != nil {
			bus.Publish(Event{
				Name: EventLLMError,
				Data: err,
			})
		}

		return err
	}

	conversation.AddMessage(response)

	if bus != nil {
		bus.Publish(Event{
			Name: EventLLMResponse,
			Data: LLMResponseEvent{
				CallID:  callID,
				Message: response,
			},
		})
	}

	return nil
}

// ConversationEngine
// TTSEngine
func NewTTSEngine(tts neurocall.TTS) *TTSEngine {
	return &TTSEngine{
		tts: tts,
	}
}
func (e *TTSEngine) SetTTS(tts neurocall.TTS) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.tts = tts
}
func (e *TTSEngine) Synthesize(
	ctx context.Context,
	text string,
) (neurocall.AudioStreamData, error) {

	if strings.TrimSpace(text) == "" {
		return neurocall.AudioStreamData{}, fmt.Errorf("text is empty")
	}

	e.mu.RLock()
	tts := e.tts
	e.mu.RUnlock()

	if tts == nil {
		return neurocall.AudioStreamData{}, fmt.Errorf(
			"TTS is not configured",
		)
	}

	return tts.Synthesize(ctx, text)
}

// TTSEngine
// TTSPlayback
func NewTTSPlayback(
	engine *TTSEngine,
) (*TTSPlayback, error) {

	if engine == nil {
		return nil, fmt.Errorf(
			"TTS engine is nil",
		)
	}

	ctx, cancel := context.WithCancel(
		context.Background(),
	)

	return &TTSPlayback{
		engine: engine,
		ctx:    ctx,
		cancel: cancel,
	}, nil
}
func (p *TTSPlayback) Close() error {
	if p == nil {
		return nil
	}

	if p.cancel != nil {
		p.cancel()
	}

	return nil
}
func (p *TTSPlayback) Synthesize(
	ctx context.Context,
	text string,
) (neurocall.AudioStreamData, error) {

	if p == nil || p.engine == nil {
		return neurocall.AudioStreamData{}, fmt.Errorf(
			"TTS playback is not configured",
		)
	}

	if ctx == nil {
		ctx = p.ctx
	}

	return p.engine.Synthesize(
		ctx,
		text,
	)
}

// TTSPlayback
// PassthroughAudio
func (PassthroughAudio) Encode(
	pcm []int16,
) []byte {
	return PCM16ToBytes(pcm)
}
func (PassthroughAudio) Decode(
	data []byte,
) []int16 {
	return BytesToPCM16(data)
}

// PassthroughAudio
// VoiceResponseEngine
func NewVoiceResponseEngine(
	tts *TTSEngine,
	bus *EventBus,
) *VoiceResponseEngine {

	engine := &VoiceResponseEngine{
		calls: make(map[string]*SIPCall),
		tts:   tts,
		bus:   bus,
	}

	if bus != nil {
		bus.Subscribe(
			EventLLMResponse,
			func(event Event) {
				data, ok := event.Data.(LLMResponseEvent)
				if !ok {
					return
				}
				if err := engine.HandleResponse(data); err != nil {
					bus.Publish(Event{
						Name: EventTTSResponseError,
						Data: TTSErrorEvent{
							CallID: data.CallID,
							Err:    err,
						},
					})
				}
			},
		)
	}
	return engine
}
func (e *VoiceResponseEngine) AddCall(
	call *SIPCall,
) error {

	if call == nil || call.CallID == "" {
		return neurocall.ErrCallNotFound
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.calls[call.CallID]; exists {
		return fmt.Errorf(
			"voice call already exists: %s",
			call.CallID,
		)
	}

	e.calls[call.CallID] = call

	return nil
}
func (e *VoiceResponseEngine) RemoveCall(
	callID string,
) error {

	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.calls[callID]; !exists {
		return neurocall.ErrCallNotFound
	}

	delete(e.calls, callID)

	return nil
}
func (e *VoiceResponseEngine) GetCall(
	callID string,
) (*SIPCall, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	call, ok := e.calls[callID]
	if !ok {
		return nil, neurocall.ErrCallNotFound
	}

	return call, nil
}
func (e *VoiceResponseEngine) HandleResponse(
	event LLMResponseEvent,
) error {
	if event.CallID == "" {
		return fmt.Errorf("call ID is empty")
	}

	text := strings.TrimSpace(event.Message.Content)
	if text == "" {
		return nil
	}

	e.mu.RLock()
	call := e.calls[event.CallID]
	e.mu.RUnlock()

	if call == nil {
		return neurocall.ErrCallNotFound
	}

	if call.Closed() {
		return neurocall.ErrCallClosed
	}

	return call.Speak(
		context.Background(),
		text,
	)
}
func (e *VoiceResponseEngine) RegisterCall(
	call *SIPCall,
) {

	if call == nil {
		return
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	e.calls[call.CallID] = call
}

// VoiceResponseEngine
// RTPPacket
func (p *RTPPacket) Marshal() ([]byte, error) {
	if p == nil {
		return nil, errInvalidRTPPacket
	}

	if p.Header.Version != 2 {
		return nil, errInvalidRTPPacket
	}

	if p.Header.CSRCCount != 0 {
		return nil, fmt.Errorf(
			"CSRC is not supported yet",
		)
	}

	data := make(
		[]byte,
		12+len(p.Payload),
	)

	data[0] = p.Header.Version << 6

	if p.Header.Padding {
		data[0] |= 1 << 5
	}

	if p.Header.Extension {
		data[0] |= 1 << 4
	}

	data[0] |= p.Header.CSRCCount & 0x0F

	data[1] = p.Header.PayloadType & 0x7F

	if p.Header.Marker {
		data[1] |= 1 << 7
	}

	binary.BigEndian.PutUint16(
		data[2:4],
		p.Header.SequenceNumber,
	)

	binary.BigEndian.PutUint32(
		data[4:8],
		p.Header.Timestamp,
	)

	binary.BigEndian.PutUint32(
		data[8:12],
		p.Header.SSRC,
	)

	copy(
		data[12:],
		p.Payload,
	)

	return data, nil
}
func (p *RTPPacket) Unmarshal(
	data []byte,
) error {
	if len(data) < 12 {
		return errInvalidRTPPacket
	}
	version := data[0] >> 6
	if version != 2 {
		return errInvalidRTPPacket
	}
	padding := data[0]&0x20 != 0
	extension := data[0]&0x10 != 0
	csrcCount := data[0] & 0x0F
	if csrcCount != 0 {
		return fmt.Errorf("CSRC is not supported yet")
	}
	header := RTPHeader{
		Version:        version,
		Padding:        padding,
		Extension:      extension,
		CSRCCount:      csrcCount,
		Marker:         data[1]&0x80 != 0,
		PayloadType:    data[1] & 0x7F,
		SequenceNumber: binary.BigEndian.Uint16(data[2:4]),
		Timestamp:      binary.BigEndian.Uint32(data[4:8]),
		SSRC:           binary.BigEndian.Uint32(data[8:12]),
	}
	offset := 12
	if extension {
		if len(data) < offset+4 {
			return errInvalidRTPPacket
		}
		extensionLength := int(binary.BigEndian.Uint16(data[offset+2:offset+4])) * 4
		offset += 4
		if len(data) < offset+extensionLength {
			return errInvalidRTPPacket
		}
		offset += extensionLength
	}
	payload := data[offset:]
	if padding {
		if len(payload) == 0 {
			return errInvalidRTPPacket
		}
		paddingLen := int(payload[len(payload)-1])
		if paddingLen == 0 || paddingLen > len(payload) {
			return errInvalidRTPPacket
		}
		payload = payload[:len(payload)-paddingLen]
	}
	p.Payload = append(
		[]byte(nil),
		payload...,
	)
	p.Header = header
	return nil
}

// RTPPacket
// RTPSession
func NewRTPSession(
	localIP string,
	port int,
	remoteIP net.IP,
	remotePort int,
	codec neurocall.Codec,
) (*RTPSession, error) {
	if !codec.Valid() {
		return nil, fmt.Errorf(
			"invalid codec",
		)
	}
	if remoteIP == nil {
		return nil, fmt.Errorf("invalid remote IP: %s", remoteIP)
	}
	localAddr := &net.UDPAddr{
		IP:   net.ParseIP(localIP),
		Port: port,
	}
	if localAddr.IP == nil {
		return nil, fmt.Errorf("invalid local IP: %s", localIP)
	}
	conn, err := net.ListenUDP("udp", localAddr)
	if err != nil {
		return nil, err
	}
	remote := &net.UDPAddr{
		IP:   remoteIP,
		Port: remotePort,
	}
	return &RTPSession{
		conn:            conn,
		remote:          remote,
		codec:           codec,
		ssrc:            randomUint32(),
		sequence:        0,
		timestamp:       0,
		sequenceTracker: &RTPSequenceTracker{},
		jitter:          NewRTPJitterBuffer(50, 30*time.Millisecond),
		writeInterval:   20 * time.Millisecond,
	}, nil
}
func (s *RTPSession) ReadOrdered() (RTPReadResult, error) {
	for {
		packet, ready := s.jitter.Pop()
		if ready {
			if packet == nil {
				return RTPReadResult{
					Lost: true,
				}, nil
			}
			return RTPReadResult{
				Packet: packet,
			}, nil
		}
		if s.jitter.HasPackets() {
			s.jitter.WaitForChange(s.jitter.maxWait)
			continue
		}
		packet, err := s.Read()
		if err != nil {
			return RTPReadResult{}, err
		}
		if err := s.jitter.Push(packet); err != nil {
			continue
		}
	}
}
func (s *RTPSession) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return nil
	}
	return conn.Close()
}
func (s *RTPSession) ReadPCM(decoder neurocall.Decoder) ([]int16, error) {

	if decoder == nil {
		return nil, fmt.Errorf("decoder is nil")
	}

	s.mu.RLock()

	if s.closed {
		s.mu.RUnlock()
		return nil, errRTPSessionClosed
	}

	codec := s.codec

	s.mu.RUnlock()

	result, err := s.ReadOrdered()

	if err != nil {
		return nil, err
	}

	if result.Lost {

		frameSamples := codec.ClockRate / 50

		if frameSamples <= 0 {
			frameSamples = 160
		}

		return concealPCM(
			codec,
			frameSamples,
		), nil
	}

	packet := result.Packet

	if packet.Header.PayloadType != codec.PayloadType {
		return nil, fmt.Errorf(
			"unexpected RTP payload type: %d",
			packet.Header.PayloadType,
		)
	}

	pcm := decoder.Decode(packet.Payload)

	if len(pcm) == 0 {
		return nil, neurocall.ErrInvalidAudio
	}

	return pcm, nil
}
func (s *RTPSession) WritePCM(encoder neurocall.Encoder, pcm []int16) error {
	if encoder == nil {
		return fmt.Errorf("encoder is nil")
	}
	if len(pcm) == 0 {
		return neurocall.ErrInvalidAudio
	}
	payload := encoder.Encode(pcm)
	if len(payload) == 0 {
		return neurocall.ErrInvalidAudio
	}
	return s.Write(payload, len(pcm))
}
func (s *RTPSession) AdvanceTimestamp(samples int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.timestamp += uint32(samples)
}
func (s *RTPSession) Stats() RTPStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.stats
}
func (s *RTPSession) ResetStats() {
	s.mu.Lock()
	s.stats = RTPStats{}
	tracker := s.sequenceTracker
	s.mu.Unlock()
	if tracker != nil {
		tracker.Reset()
	}
}
func (s *RTPSession) Read() (*RTPPacket, error) {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return nil, errRTPSessionClosed
	}
	conn := s.conn
	s.mu.RUnlock()
	if conn == nil {
		return nil, errRTPSessionClosed
	}
	buf := make([]byte, 2048)
	n, _, err := conn.ReadFromUDP(buf)
	if err != nil {
		s.mu.RLock()
		closed := s.closed
		s.mu.RUnlock()
		if closed {
			return nil, errRTPSessionClosed
		}
		return nil, err
	}
	s.mu.Lock()
	s.stats.BytesReceived += uint64(n)
	s.mu.Unlock()
	packet := &RTPPacket{}
	if err := packet.Unmarshal(buf[:n]); err != nil {
		s.mu.Lock()
		s.stats.InvalidPackets++
		s.mu.Unlock()
		return nil, err
	}
	s.sequenceTracker.Update(packet.Header.SequenceNumber)
	trackerStats := s.sequenceTracker.Stats()
	s.mu.Lock()
	s.stats.ReceivedPackets++
	s.stats.LostPackets = trackerStats.LostPackets
	s.mu.Unlock()
	return packet, nil
}
func (s *RTPSession) Write(payload []byte, samples int) error {

	if len(payload) == 0 {
		return neurocall.ErrInvalidAudio
	}

	if samples <= 0 {
		return neurocall.ErrInvalidAudio
	}

	s.mu.Lock()

	if s.closed {
		s.mu.Unlock()
		return errRTPSessionClosed
	}

	conn := s.conn
	remote := s.remote

	packet := RTPPacket{
		Header: RTPHeader{
			Version:        2,
			PayloadType:    s.codec.PayloadType,
			SequenceNumber: s.sequence,
			Timestamp:      s.timestamp,
			SSRC:           s.ssrc,
		},
		Payload: append([]byte(nil), payload...),
	}

	data, err := packet.Marshal()

	if err != nil {
		s.mu.Unlock()
		return err
	}

	s.sequence++
	s.timestamp += uint32(samples)

	s.mu.Unlock()

	if _, err := conn.WriteToUDP(
		data,
		remote,
	); err != nil {
		return err
	}

	s.mu.Lock()

	s.stats.SentPackets++
	s.stats.BytesSent += uint64(len(data))

	s.mu.Unlock()

	return nil
}
func (s *RTPSession) WritePCMRealtime(ctx context.Context, encoder neurocall.Encoder, pcm []int16) error {

	if encoder == nil {
		return fmt.Errorf("encoder is nil")
	}

	if len(pcm) == 0 {
		return neurocall.ErrInvalidAudio
	}
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.RLock()

	interval := s.writeInterval

	s.mu.RUnlock()

	if interval <= 0 {
		interval = 20 * time.Millisecond
	}

	sampleRate := s.codec.ClockRate

	if sampleRate <= 0 {
		sampleRate = 8000
	}

	frameSize := sampleRate / 50

	if frameSize <= 0 {
		frameSize = 160
	}

	ticker := time.NewTicker(interval)

	defer ticker.Stop()

	first := true

	for offset := 0; offset < len(pcm); offset += frameSize {

		end := offset + frameSize

		if end > len(pcm) {
			end = len(pcm)
		}

		frame := pcm[offset:end]

		if !first {
			select {
			case <-ctx.Done():
				return ctx.Err()

			case <-ticker.C:
			}
		}

		first = false

		if err := s.WritePCM(
			encoder,
			frame,
		); err != nil {
			return err
		}
	}

	return nil
}

// RTPSession
// RTPSequenceTracker
func NewRTPSequenceTracker() *RTPSequenceTracker {
	return &RTPSequenceTracker{}
}
func (t *RTPSequenceTracker) Update(seq uint16) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.started {
		t.started = true
		t.last = seq
		t.received++
		return
	}

	// Same sequence = duplicate.
	if seq == t.last {
		t.duplicate++
		return
	}

	diff := uint16(seq - t.last)

	// Forward packet.
	//
	// This also correctly handles:
	//
	// 65534 -> 65535
	// 65535 -> 0
	// 0 -> 1
	//
	if diff < 0x8000 {
		if diff > 1 {
			t.lost += uint64(diff - 1)
		}

		t.last = seq
		t.received++
		return
	}

	// Packet arrived behind current sequence.
	t.outOfOrder++
}
func (t *RTPSequenceTracker) Stats() RTPStats {
	t.mu.Lock()
	defer t.mu.Unlock()

	return RTPStats{
		ReceivedPackets: t.received,
		LostPackets:     t.lost,
		Duplicates:      t.duplicate,
		OutOfOrder:      t.outOfOrder,

		LastSequence: t.last,
		HasSequence:  t.started,
	}
}
func (t *RTPSequenceTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	*t = RTPSequenceTracker{}
}

// RTPSequenceTracker
// RTPJitterBuffer
func NewRTPJitterBuffer(
	maxPackets int,
	maxWait time.Duration,
) *RTPJitterBuffer {

	if maxPackets <= 0 {
		maxPackets = 50
	}

	if maxWait <= 0 {
		maxWait = 30 * time.Millisecond
	}

	return &RTPJitterBuffer{
		packets:    make(map[uint16]jitterPacket),
		maxPackets: maxPackets,
		maxWait:    maxWait,
		notify:     make(chan struct{}, 1),
	}
}
func (b *RTPJitterBuffer) Push(packet *RTPPacket) error {
	if packet == nil {
		return errInvalidRTPPacket
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	seq := packet.Header.SequenceNumber
	if _, exists := b.packets[seq]; exists {
		return nil
	}
	// Before the buffer has produced anything, allow
	// reordering around the initial sequence number.
	if !b.started {
		b.started = true
		b.next = seq
	}
	// Once output has started, packets behind next are stale.
	if b.advanced && seqLess(seq, b.next) {
		return nil
	}
	b.packets[seq] = jitterPacket{
		packet:     packet,
		receivedAt: time.Now(),
	}
	select {
	case b.notify <- struct{}{}:
	default:
	}
	// Before output starts, an earlier packet becomes
	// the new starting point.
	if !b.advanced && seqLess(seq, b.next) {
		b.next = seq
	}

	if len(b.packets) > b.maxPackets {
		b.dropOldestLocked()
	}

	return nil
}
func (b *RTPJitterBuffer) Pop() (*RTPPacket, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.started {
		return nil, false
	}

	// Expected packet is available.
	if entry, ok := b.packets[b.next]; ok {
		delete(b.packets, b.next)
		b.next++
		b.advanced = true

		return entry.packet, true
	}

	// No packet after the expected sequence.
	if len(b.packets) == 0 {
		return nil, false
	}

	// Something later exists. Check whether the missing
	// packet has waited long enough.
	var oldestWait time.Duration

	for _, entry := range b.packets {
		wait := time.Since(entry.receivedAt)

		if wait > oldestWait {
			oldestWait = wait
		}
	}

	if oldestWait < b.maxWait {
		return nil, false
	}

	// Expected packet is considered lost.
	b.next++
	b.advanced = true

	return nil, true
}
func (b *RTPJitterBuffer) dropOldestLocked() {
	if len(b.packets) == 0 {
		return
	}
	// The oldest packet is the one closest to next
	// in forward RTP sequence space.
	oldest := b.next
	found := false
	var oldestDistance uint16
	for seq := range b.packets {
		distance := seqDistance(b.next, seq)
		if !found || distance < oldestDistance {
			oldest = seq
			oldestDistance = distance
			found = true
		}
	}
	if !found {
		return
	}
	delete(b.packets, oldest)
	// If we dropped the packet we were waiting for,
	// advance next so Pop() can continue.
	if oldest == b.next {
		b.next++
	}
}
func (b *RTPJitterBuffer) WaitForChange(timeout time.Duration) {
	if timeout <= 0 {
		timeout = b.maxWait
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-b.notify:
	case <-timer.C:
	}
}
func (b *RTPJitterBuffer) HasPackets() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.packets) > 0
}

// RTPJitterBuffer
// FakeSTT
func NewFakeSTT() *FakeSTT {
	return &FakeSTT{}
}
func (f *FakeSTT) Transcribe(
	ctx context.Context,
	segment neurocall.AudioSegment,
) (neurocall.Transcript, error) {
	select {
	case <-ctx.Done():
		return neurocall.Transcript{}, ctx.Err()
	default:
	}

	if len(segment.Data) == 0 {
		return neurocall.Transcript{}, neurocall.ErrInvalidAudio
	}

	return neurocall.Transcript{
		Text:       "hello from fake stt",
		Confidence: 0.99,
		Language:   "en",
	}, nil
}

// FakeSTT
// AudioReceiveConfig
func DefaultAudioReceiveConfig() AudioReceiveConfig {
	return AudioReceiveConfig{
		SampleRate: 8000,
		Channels:   1,
		FrameSize:  160,
	}
}

// AudioReceiveConfig
// AudioProcessor
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
