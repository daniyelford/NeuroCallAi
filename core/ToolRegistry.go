package core

import (
	"fmt"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		tools: make(map[string]neurocall.Tool),
	}
}
func (r *ToolRegistry) Register(
	tool neurocall.Tool,
) error {
	if tool == nil {
		return neurocall.ErrToolNotFound
	}
	name := tool.Name()
	if name == "" {
		return fmt.Errorf(
			"tool name is empty",
		)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[name]; exists {
		return fmt.Errorf(
			"tool already registered: %s",
			name,
		)
	}
	r.tools[name] = tool
	return nil
}
func (r *ToolRegistry) Get(
	name string,
) (neurocall.Tool, bool) {

	r.mu.RLock()
	defer r.mu.RUnlock()

	tool, ok := r.tools[name]

	return tool, ok
}
func (r *ToolRegistry) List() []neurocall.Tool {

	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(
		[]neurocall.Tool,
		0,
		len(r.tools),
	)

	for _, tool := range r.tools {
		result = append(result, tool)
	}

	return result
}
func (r *ToolRegistry) Call(
	name string,
	args map[string]any,
) (any, error) {

	tool, ok := r.Get(name)

	if !ok {
		return nil, neurocall.ErrToolNotFound
	}

	return tool.Call(args)
}
