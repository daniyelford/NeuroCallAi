package plugin

import (
	"context"
	"errors"
	"fmt"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

func NewRegistry() *Registry {

	return &Registry{
		plugins: make(map[string]neurocall.Plugin),
	}

}
func WithPlugin(p Plugin) Option {

	return func(a *App) {

		_ = a.registry.Register(p)

	}

}
func (r *Registry) Register(
	p neurocall.Plugin,
) error {

	if p == nil {
		return ErrNilPlugin
	}

	name := p.Name()

	if name == "" {
		return ErrEmptyPluginName
	}

	if _, exists := r.plugins[name]; exists {
		return fmt.Errorf(
			"%w: %s",
			ErrDuplicatePlugin,
			name,
		)
	}

	r.plugins[name] = p

	return nil
}

func (r *Registry) Get(
	name string,
) (neurocall.Plugin, bool) {

	p, ok := r.plugins[name]

	return p, ok
}
func NewGraph(registry *Registry) *Graph {
	return &Graph{
		registry: registry,
	}
}

func (g *Graph) Resolve() (
	[]neurocall.Plugin,
	error,
) {
	plugins := g.registry.List()

	state := make(map[string]uint8)

	result := make([]neurocall.Plugin, 0, len(plugins))

	var visit func(neurocall.Plugin) error

	visit = func(p neurocall.Plugin) error {

		name := p.Name()

		switch state[name] {
		case 1:
			return ErrDependencyCycle

		case 2:
			return nil
		}

		state[name] = 1

		for _, dependency := range p.DependsOn() {

			dep, ok := g.registry.Get(dependency)

			if !ok {
				return ErrDependencyNotFound
			}

			if err := visit(dep); err != nil {
				return err
			}
		}

		state[name] = 2

		result = append(result, p)

		return nil
	}

	for _, p := range plugins {

		if err := visit(p); err != nil {
			return nil, err
		}
	}

	return result, nil
}
func (r *Registry) List() []neurocall.Plugin {

	result := make(
		[]neurocall.Plugin,
		0,
		len(r.plugins),
	)

	for _, p := range r.plugins {
		result = append(result, p)
	}

	return result
}
func NewManager() *Manager {
	return &Manager{
		registry: NewRegistry(),
	}
}

func (m *Manager) Register(
	p neurocall.Plugin,
) error {
	return m.registry.Register(p)
}

// func (m *Manager) Init(
// 	ctx *neurocall.Context,
// ) error {

// 	order, err := NewGraph(m.registry).Resolve()

// 	if err != nil {
// 		return err
// 	}

// 	m.order = order

// 	for _, p := range m.order {

// 		if err := p.Init(ctx); err != nil {
// 			return err
// 		}
// 	}

//		return nil
//	}
func (m *Manager) Init(
	ctx *neurocall.Context,
) error {

	order, err := NewGraph(m.registry).Resolve()

	if err != nil {
		return err
	}

	m.order = order

	initialized := make(
		[]neurocall.Plugin,
		0,
		len(order),
	)

	for _, p := range order {

		if err := p.Init(ctx); err != nil {

			for i := len(initialized) - 1; i >= 0; i-- {
				_ = initialized[i].Stop()
			}

			return err
		}

		initialized = append(
			initialized,
			p,
		)
	}

	return nil
}
func (m *Manager) Start() error {

	for _, p := range m.order {

		if err := p.Start(); err != nil {
			return err
		}
	}

	return nil
}

func (m *Manager) Stop() error {

	for i := len(m.order) - 1; i >= 0; i-- {

		if err := m.order[i].Stop(); err != nil {
			return err
		}
	}

	return nil
}

func (p *failingPlugin) Name() string {
	return p.name
}

func (p *failingPlugin) DependsOn() []string {
	return nil
}

func (p *failingPlugin) Init(
	_ *neurocall.Context,
) error {
	return errors.New("init failed")
}

func (p *failingPlugin) Start() error {
	return nil
}

func (p *failingPlugin) Stop() error {
	return nil
}

func NewPipeline(
	source neurocall.AudioSource,
	processor neurocall.AudioProcessor,
	sink neurocall.AudioSink,
) *Pipeline {

	return &Pipeline{
		source:    source,
		processor: processor,
		sink:      sink,
	}
}

func (p *Pipeline) Run(
	ctx context.Context,
) error {

	for {

		frame, err := p.source.Read(ctx)

		if err != nil {
			return err
		}

		frame, err = p.processor.Process(
			ctx,
			frame,
		)

		if err != nil {
			return err
		}

		if err := p.sink.Write(
			ctx,
			frame,
		); err != nil {
			return err
		}
	}
}
