package core

import "fmt"

func NewPluginRegistry() *PluginRegistry {
	return &PluginRegistry{
		plugins: make(map[string]Plugin),
	}
}
func (r *PluginRegistry) Register(
	plugin Plugin,
) error {
	if plugin == nil {
		return fmt.Errorf(
			"cannot register nil plugin",
		)
	}

	name := plugin.Name()

	if name == "" {
		return fmt.Errorf(
			"plugin name is empty",
		)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.plugins[name]; exists {
		return fmt.Errorf(
			"plugin already registered: %s",
			name,
		)
	}

	r.plugins[name] = plugin

	return nil
}
func (r *PluginRegistry) Get(
	name string,
) (Plugin, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	plugin, ok := r.plugins[name]

	return plugin, ok
}
func (r *PluginRegistry) List() []Plugin {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(
		[]Plugin,
		0,
		len(r.plugins),
	)

	for _, plugin := range r.plugins {
		result = append(result, plugin)
	}

	return result
}
