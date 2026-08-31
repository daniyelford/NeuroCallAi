package core

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
