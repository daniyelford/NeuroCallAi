package core

import "github.com/daniyelford/NeuroCallAi/pkg/neurocall"

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
