package core

func NewPortAllocator(min, max int) *PortAllocator {
	if min%2 != 0 {
		min++
	}
	available := make(map[int]struct{})
	for port := min; port <= max; port += 2 {
		available[port] = struct{}{}
	}
	return &PortAllocator{
		min:       min,
		max:       max,
		available: available,
	}
}
func (a *PortAllocator) Allocate() (int, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for port := a.min; port <= a.max; port += 2 {
		if _, ok := a.available[port]; ok {
			delete(a.available, port)
			return port, nil
		}
	}
	return 0, errRTPPortExhausted
}
func (a *PortAllocator) Release(port int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if port < a.min || port > a.max || port%2 != 0 {
		return
	}
	a.available[port] = struct{}{}
}
