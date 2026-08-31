package core

import (
	"fmt"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

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

		bus: NewEventBus(),
	}
}
func (m *CallManager) SetEventBus(bus *EventBus) {
	if m == nil || bus == nil {
		return
	}
	m.mu.Lock()
	m.bus = bus
	m.mu.Unlock()
}
func (m *CallManager) StartSession(call *SIPCall) (*CallSession, error) {
	session, err := m.CreateSession(call)
	if err != nil {
		return nil, err
	}
	if err := session.Start(); err != nil {
		_ = m.RemoveSession(call.CallID)
		return nil, err
	}
	return session, nil
}
func (m *CallManager) Add(call *SIPCall) error {

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
func (m *CallManager) Get(id string) (*SIPCall, error) {
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

	var localPort int
	if call != nil {
		call.mu.Lock()
		localPort = call.LocalRTPPort
		call.LocalRTPPort = 0
		call.mu.Unlock()
	}
	if session != nil {
		_ = session.Close()
	}
	if call != nil {
		_ = call.CloseResources()
	}
	if localPort > 0 {
		m.ports.Release(localPort)
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
func (m *CallManager) CreateSession(call *SIPCall) (*CallSession, error) {
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
	bus := m.bus
	if bus == nil {
		bus = NewEventBus()
		m.bus = bus
	}
	session, err := NewCallSession(
		call,
		bus,
	)
	if err != nil {
		return nil, err
	}
	m.sessions[call.CallID] = session
	return session, nil
}
func (m *CallManager) GetSession(id string) (*CallSession, error) {

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
