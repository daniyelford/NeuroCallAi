package event

import (
	"sync/atomic"
	"testing"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

func TestPublishSubscribe(t *testing.T) {

	bus := NewBus()

	var count atomic.Int32

	bus.Subscribe(
		"test",
		func(e neurocall.Event) {
			count.Add(1)
		},
	)

	bus.Publish(
		neurocall.NewEvent("test"),
	)

	if count.Load() != 1 {
		t.Fatalf(
			"expected 1 handler call, got %d",
			count.Load(),
		)
	}
}

func TestUnsubscribe(t *testing.T) {

	bus := NewBus()

	var count atomic.Int32

	sub := bus.Subscribe(
		"test",
		func(e neurocall.Event) {
			count.Add(1)
		},
	)

	sub.Unsubscribe()

	bus.Publish(
		neurocall.NewEvent("test"),
	)

	if count.Load() != 0 {
		t.Fatalf(
			"expected 0 handler calls, got %d",
			count.Load(),
		)
	}
}
