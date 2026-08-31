package core

func NewShutdownManager() *ShutdownManager {
	return &ShutdownManager{}
}
func (s *ShutdownManager) Register(
	handler func() error,
) {
	if handler == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.handlers =
		append(s.handlers, handler)
}
func (s *ShutdownManager) Shutdown() error {
	s.mu.Lock()

	if s.shuttingDown {
		s.mu.Unlock()
		return nil
	}

	s.shuttingDown = true

	handlers := append(
		[]func() error(nil),
		s.handlers...,
	)

	s.mu.Unlock()

	var firstErr error

	for _, handler := range handlers {
		if err := handler(); err != nil &&
			firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}
