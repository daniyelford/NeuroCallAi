package bootstrap

import (
	"github.com/daniyelford/neurocall/internal/config"
	"github.com/daniyelford/neurocall/internal/container"
	"github.com/daniyelford/neurocall/internal/event"
	"github.com/daniyelford/neurocall/internal/logger"
	"github.com/daniyelford/neurocall/internal/plugin"
	"github.com/daniyelford/neurocall/pkg/neurocall"
)

// func NewApp() *neurocall.App {

// 	logging := logger.NewConsole()
// 	cfg := config.NewMemory()
// 	events := event.NewBus()
// 	di := container.New()
// 	plugins := plugin.NewManager()

//		return neurocall.New(
//			plugins,
//			events,
//			logging,
//			cfg,
//			di,
//		)
//	}
func NewApp() *neurocall.App {

	logging := logger.NewConsole()
	cfg := config.NewMemory()
	events := event.NewBus()
	di := container.New()
	plugins := plugin.NewManager()

	_ = di.Register("logger", logging)
	_ = di.Register("config", cfg)
	_ = di.Register("events", events)

	return neurocall.New(
		plugins,
		events,
		logging,
		cfg,
		di,
	)
}
