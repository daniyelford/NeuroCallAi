package core

func NewPluginGraph() *PluginGraph {
	return &PluginGraph{
		dependencies: make(
			map[string][]string,
		),
	}
}
func (g *PluginGraph) AddDependency(
	plugin string,
	dependency string,
) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.dependencies[plugin] =
		append(
			g.dependencies[plugin],
			dependency,
		)
}
func (g *PluginGraph) Dependencies(
	plugin string,
) []string {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return append(
		[]string(nil),
		g.dependencies[plugin]...,
	)
}
