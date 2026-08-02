package event

import (
	"sync"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

type Bus struct {
	mu       sync.RWMutex
	nextID   uint64
	handlers map[string]map[uint64]neurocall.EventHandler
}
type subscription struct {
	bus  *Bus
	name string
	id   uint64

	once sync.Once
}
