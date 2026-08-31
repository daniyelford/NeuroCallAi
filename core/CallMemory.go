package core

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
