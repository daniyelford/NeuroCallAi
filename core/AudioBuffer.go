package core

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
