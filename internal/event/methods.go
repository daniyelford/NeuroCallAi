package event

import (
	"github.com/daniyelford/neurocall/pkg/neurocall"
)

func NewBus() *Bus {
	return &Bus{
		handlers: make(
			map[string]map[uint64]neurocall.EventHandler,
		),
	}
}
func (b *Bus) Publish(e neurocall.Event) {
	if e == nil {
		return
	}

	b.mu.RLock()

	handlers := make(
		[]neurocall.EventHandler,
		0,
		len(b.handlers[e.Name()]),
	)

	for _, handler := range b.handlers[e.Name()] {
		handlers = append(handlers, handler)
	}

	b.mu.RUnlock()

	for _, handler := range handlers {
		handler(e)
	}
}

func (b *Bus) Subscribe(
	name string,
	handler neurocall.EventHandler,
) neurocall.Subscription {

	b.mu.Lock()
	defer b.mu.Unlock()

	b.nextID++

	id := b.nextID

	if b.handlers[name] == nil {
		b.handlers[name] =
			make(map[uint64]neurocall.EventHandler)
	}

	b.handlers[name][id] = handler

	return &subscription{
		bus:  b,
		name: name,
		id:   id,
	}
}

func (s *subscription) Unsubscribe() {
	s.once.Do(func() {
		s.bus.remove(
			s.name,
			s.id,
		)
	})
}
func (b *Bus) unsubscribe(
	name string,
	handler neurocall.EventHandler,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	handlers := b.handlers[name]

	for i, current := range handlers {
		if handlerEqual(current, handler) {
			b.handlers[name] = append(
				handlers[:i],
				handlers[i+1:]...,
			)

			break
		}
	}
}

func handlerEqual(
	a neurocall.EventHandler,
	b neurocall.EventHandler,
) bool {
	return &a != nil && &b != nil
}
func (b *Bus) remove(
	name string,
	id uint64,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	handlers, ok := b.handlers[name]

	if !ok {
		return
	}

	delete(handlers, id)

	if len(handlers) == 0 {
		delete(b.handlers, name)
	}
}
