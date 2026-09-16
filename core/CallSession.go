package core

import (
	"context"
	"fmt"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewCallSession(call *SIPCall, bus *EventBus) (*CallSession, error) {
	if call == nil {
		return nil, neurocall.ErrCallNotFound
	}
	if call.CallID == "" {
		return nil, neurocall.ErrCallNotFound
	}
	ctx, cancel := context.WithCancel(
		context.Background(),
	)
	if bus == nil {
		bus = NewEventBus()
	}
	conversation := NewConversation()
	session := &CallSession{
		ID:           call.CallID,
		Call:         call,
		Events:       bus,
		Conversation: conversation,
		Memory:       NewMemory(),
		Tools:        NewToolRegistry(),
		ctx:          ctx,
		cancel:       cancel,
	}
	return session, nil
}
func (s *CallSession) ConfigureConversation(llm *LLMEngine) error {
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
	conversation := s.Conversation
	bus := s.Events
	callID := s.ID
	engine := NewConversationEngine(llm, bus)
	s.LLM = llm
	s.ConversationEngine = engine
	s.mu.Unlock()
	if conversation == nil {
		return fmt.Errorf("conversation is not configured")
	}
	if err := engine.SetConversation(callID, conversation); err != nil {
		return err
	}
	return s.AttachSTTEvents()
}
func (s *CallSession) ConfigureVoice(tts *TTSEngine) error {
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
	voice := NewVoiceResponseEngine(tts, s.Events)
	s.VoiceEngine = voice
	call := s.Call
	s.mu.Unlock()
	if call != nil {
		voice.RegisterCall(call)
	}
	return nil
}
func (s *CallSession) ConfigureSTT(worker *STTWorker) error {
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
	callID := s.ID
	s.sttEventsAttached = true
	s.mu.Unlock()
	bus.Subscribe(
		EventTranscript,
		func(event Event) {
			data, ok := event.Data.(TranscriptEvent)
			if !ok {
				return
			}
			if data.CallID != callID {
				return
			}
			if s.Closed() {
				return
			}
			go func() {
				if s.Closed() {
					return
				}
				if err := engine.HandleTranscript(
					data,
				); err != nil {
					if s.Closed() {
						return
					}
					bus.Publish(Event{
						Name: EventLLMError,
						Data: err,
					})
				}
			}()
		},
	)
	return nil
}
func (s *CallSession) ProcessAudioSegment(segment neurocall.AudioSegment) error {
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
func (s *CallSession) ProcessAudioFrame(ctx context.Context, frame neurocall.AudioFrame, pcm []int16) error {
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
func (s *CallSession) SetSTT(stt neurocall.STT) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.STT = stt
}
func (s *CallSession) SetStreamingSTT(stt neurocall.StreamingSTT) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.StreamingSTT = stt
}
func (s *CallSession) SetTTS(tts neurocall.TTS) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.TTS = tts
}
func (s *CallSession) SetLLM(llm neurocall.LLM) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.LLM = llm
}
func (s *CallSession) SetMemory(memory neurocall.Memory) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Memory = memory
}
func (s *CallSession) SetTools(tools neurocall.ToolRegistry) {
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
		ID:       s.ID,
		Running:  s.running,
		Closed:   s.closed,
		HasAudio: s.Audio != nil,
		HasSTT:   s.STTWorker != nil,
		HasLLM:   s.ConversationEngine != nil,
		HasTTS:   s.VoiceEngine != nil,
	}
}
func (s *CallSession) PushAudio(audio []int16) error {
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
func (s *CallSession) PushSTTSegment(segment neurocall.AudioSegment) error {
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
	conversationEngine := s.ConversationEngine
	callID := s.ID
	s.STTWorker = nil
	s.Audio = nil
	s.cancel = nil
	s.mu.Unlock()
	if conversationEngine != nil {
		_ = conversationEngine.Remove(callID)
	}
	if worker != nil {
		worker.Stop()
	}
	if audio != nil {
		_ = audio.Close()
	}
	if cancel != nil {
		cancel()
	}
	if events != nil {
		events.Publish(Event{
			Name: EventCallEnded,
			Data: s,
		})
	}
	return nil
}
