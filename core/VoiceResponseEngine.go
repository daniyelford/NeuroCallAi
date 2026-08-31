package core

import (
	"context"
	"fmt"
	"strings"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

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
