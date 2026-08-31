package core

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
