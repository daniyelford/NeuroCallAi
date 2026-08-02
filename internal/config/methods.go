package config

import "os"

func NewMemory() *Memory {
	return &Memory{
		values: make(map[string]string),
	}
}

func (c *Memory) Set(key, value string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.values[key] = value
}

func (c *Memory) Get(key string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	value, ok := c.values[key]

	return value, ok
}
func LoadYAML(path string) (*Config, error) {

	data, err := os.ReadFile(path)

	if err != nil {
		return nil, err
	}

	var cfg Config

	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}
