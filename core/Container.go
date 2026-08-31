package core

func NewContainer() *Container {
	return &Container{
		services: make(map[string]any),
	}
}
func (c *Container) Set(
	name string,
	service any,
) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.services[name] = service
}
func (c *Container) Get(
	name string,
) (any, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	value, ok := c.services[name]

	return value, ok
}
