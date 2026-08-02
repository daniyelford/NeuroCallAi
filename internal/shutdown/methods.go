package shutdown

import (
	"context"
	"os"
	"os/signal"
	"syscall"
)

func (m *Manager) Register(hook Hook) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.hooks = append(m.hooks, hook)
}

func (m *Manager) Run(ctx context.Context) error {
	m.mu.Lock()

	hooks := append(
		[]Hook(nil),
		m.hooks...,
	)

	m.mu.Unlock()

	for i := len(hooks) - 1; i >= 0; i-- {
		if err := hooks[i](ctx); err != nil {
			return err
		}
	}

	return nil
}
func Wait(ctx context.Context) error {

	ctx, cancel := signal.NotifyContext(
		ctx,
		os.Interrupt,
		syscall.SIGTERM,
	)

	defer cancel()

	<-ctx.Done()

	return ctx.Err()
}
func NewSignalHandler() *SignalHandler {
	return &SignalHandler{
		signals: []os.Signal{
			os.Interrupt,
			syscall.SIGTERM,
		},
	}
}

func (h *SignalHandler) Wait(ctx context.Context) error {
	ctx, stop := signal.NotifyContext(
		ctx,
		h.signals...,
	)
	defer stop()

	<-ctx.Done()

	return ctx.Err()
}
