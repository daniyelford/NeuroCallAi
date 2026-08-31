package core

import (
	"fmt"
	"strings"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

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
func (e *ConversationEngine) SetConversation(
	callID string,
	conversation *Conversation,
) error {
	if e == nil {
		return fmt.Errorf("conversation engine is nil")
	}

	if callID == "" {
		return fmt.Errorf("call ID is empty")
	}

	if conversation == nil {
		return fmt.Errorf("conversation is nil")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if e.conversations == nil {
		e.conversations = make(map[string]*Conversation)
	}

	e.conversations[callID] = conversation

	return nil
}
func (e *ConversationEngine) GetOrCreate(
	callID string,
) (*Conversation, error) {
	if e == nil {
		return nil, fmt.Errorf("conversation engine is nil")
	}
	if callID == "" {
		return nil, fmt.Errorf("call ID is empty")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.conversations == nil {
		e.conversations = make(map[string]*Conversation)
	}
	if conversation, exists := e.conversations[callID]; exists {
		return conversation, nil
	}
	conversation := NewConversation()
	e.conversations[callID] = conversation
	return conversation, nil
}
func (e *ConversationEngine) Remove(callID string) error {
	if e == nil {
		return fmt.Errorf("conversation engine is nil")
	}

	if callID == "" {
		return fmt.Errorf("call ID is empty")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.conversations[callID]; !exists {
		return fmt.Errorf(
			"conversation not found: %s",
			callID,
		)
	}

	delete(e.conversations, callID)

	return nil
}
func (e *ConversationEngine) HandleTranscript(
	event TranscriptEvent,
) error {
	if e == nil {
		return fmt.Errorf("conversation engine is nil")
	}
	if event.CallID == "" {
		return fmt.Errorf("call ID is empty")
	}
	text := strings.TrimSpace(event.Transcript.Text)
	if text == "" {
		return nil
	}
	conversation, err := e.GetOrCreate(event.CallID)
	if err != nil {
		return err
	}
	conversation.responseMu.Lock()
	defer conversation.responseMu.Unlock()
	conversation.AddMessage(neurocall.Message{
		Role:    "user",
		Content: text,
	})
	return e.generateResponseLocked(
		event.CallID,
		conversation,
	)
}
func (e *ConversationEngine) generateResponseLocked(
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
