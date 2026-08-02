package plugin

import (
	"context"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

type Registry struct {
	plugins map[string]neurocall.Plugin
}
type Manager struct {
	registry *Registry
	order    []neurocall.Plugin
}
type Plugin interface {
	Name() string

	DependsOn() []string

	Init(*Context) error
	Start() error
	Stop() error
}
type Engine struct {
	runtime Runtime

	manager PluginManager
}
type Graph struct {
	registry *Registry
}
type failingPlugin struct {
	name string
}
type AudioStream interface {
	Read(
		context.Context,
	) (AudioFrame, error)

	Write(
		context.Context,
		AudioFrame,
	) error

	Close() error
}
