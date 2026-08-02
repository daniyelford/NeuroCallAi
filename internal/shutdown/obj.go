package shutdown

import (
	"context"
	"os"
	"sync"
)

type Hook func(context.Context) error

type Manager struct {
	mu    sync.Mutex
	hooks []Hook
}
type SignalHandler struct {
	signals []os.Signal
}
