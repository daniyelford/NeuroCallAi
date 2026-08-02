package plugin

import (
	"testing"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

type testPlugin struct {
	name string
	deps []string
}

func (p *testPlugin) Name() string {
	return p.name
}

func (p *testPlugin) DependsOn() []string {
	return p.deps
}

func (p *testPlugin) Init(
	_ *neurocall.Context,
) error {
	return nil
}

func (p *testPlugin) Start() error {
	return nil
}

func (p *testPlugin) Stop() error {
	return nil
}

func TestRegistryDuplicate(t *testing.T) {

	r := NewRegistry()

	p := &testPlugin{name: "logger"}

	if err := r.Register(p); err != nil {
		t.Fatal(err)
	}

	err := r.Register(p)

	if err == nil {
		t.Fatal("expected duplicate plugin error")
	}
}

func TestDependencyResolution(t *testing.T) {

	r := NewRegistry()

	config := &testPlugin{
		name: "config",
	}

	logger := &testPlugin{
		name: "logger",
		deps: []string{"config"},
	}

	mongo := &testPlugin{
		name: "mongo",
		deps: []string{"logger"},
	}

	if err := r.Register(mongo); err != nil {
		t.Fatal(err)
	}

	if err := r.Register(logger); err != nil {
		t.Fatal(err)
	}

	if err := r.Register(config); err != nil {
		t.Fatal(err)
	}

	order, err := NewGraph(r).Resolve()

	if err != nil {
		t.Fatal(err)
	}

	expected := []string{
		"config",
		"logger",
		"mongo",
	}

	for i, p := range order {

		if p.Name() != expected[i] {
			t.Fatalf(
				"expected %s, got %s",
				expected[i],
				p.Name(),
			)
		}
	}
}

func TestDependencyCycle(t *testing.T) {

	r := NewRegistry()

	a := &testPlugin{
		name: "a",
		deps: []string{"b"},
	}

	b := &testPlugin{
		name: "b",
		deps: []string{"a"},
	}

	if err := r.Register(a); err != nil {
		t.Fatal(err)
	}

	if err := r.Register(b); err != nil {
		t.Fatal(err)
	}

	_, err := NewGraph(r).Resolve()

	if err != ErrDependencyCycle {
		t.Fatalf(
			"expected dependency cycle, got %v",
			err,
		)
	}
}
