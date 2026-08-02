package container

import (
	"errors"
	"sync"
)

var (
	ErrAlreadyRegistered = errors.New(
		"dependency already registered",
	)

	ErrNotFound = errors.New(
		"dependency not found",
	)
)

type Container struct {
	mu     sync.RWMutex
	values map[string]any
}

func New() *Container {
	return &Container{
		values: make(map[string]any),
	}
}

func (c *Container) Register(
	name string,
	value any,
) error {

	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.values[name]; exists {
		return ErrAlreadyRegistered
	}

	c.values[name] = value

	return nil
}

func (c *Container) Resolve(
	name string,
) (any, error) {

	c.mu.RLock()
	defer c.mu.RUnlock()

	value, ok := c.values[name]

	if !ok {
		return nil, ErrNotFound
	}

	return value, nil
}
