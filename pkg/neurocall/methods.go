package neurocall

import (
	"context"
	"time"
)

//	func New(opts ...Option) *App {
//		app := &App{
//			state: Created,
//		}
//		for _, opt := range opts {
//			opt(app)
//		}
//		app.ctx.App = app
//		for _, opt := range opts {
//			opt(app)
//		}
//		return app
//	}
func NewEvent(name string) BasicEvent {
	return BasicEvent{
		EventName: name,
		CreatedAt: time.Now(),
	}
}

func (e BasicEvent) Name() string {
	return e.EventName
}

func (e BasicEvent) Time() time.Time {
	return e.CreatedAt
}

func (s State) String() string {
	switch s {
	case Created:
		return "created"
	case Initializing:
		return "initializing"
	case Starting:
		return "starting"
	case Running:
		return "running"
	case Stopping:
		return "stopping"
	case Stopped:
		return "stopped"
	default:
		return "unknown"
	}
}

// func New(
// 	plugins PluginManager,
// 	events EventBus,
// ) *App {

// 	app := &App{
// 		state:   Created,
// 		plugins: plugins,
// 		events:  events,
// 	}

// 	app.ctx = Context{
// 		App:    app,
// 		Events: events,
// 	}

// 	return app
// }

func (a *App) Run() error {

	a.mu.Lock()

	if a.state != Created {
		a.mu.Unlock()

		return ErrInvalidState
	}

	a.state = Initializing

	a.mu.Unlock()

	if err := a.plugins.Init(&a.ctx); err != nil {
		return err
	}

	a.mu.Lock()
	a.state = Starting
	a.mu.Unlock()

	if err := a.plugins.Start(); err != nil {
		return err
	}

	a.mu.Lock()
	a.state = Running
	a.mu.Unlock()

	return nil
}

func (a *App) Shutdown(ctx context.Context) error {

	a.mu.Lock()

	if a.state == Stopped {
		a.mu.Unlock()

		return nil
	}

	a.state = Stopping

	a.mu.Unlock()

	err := a.plugins.Stop()

	a.mu.Lock()
	a.state = Stopped
	a.mu.Unlock()

	return err
}

func (a *App) State() State {
	a.mu.RLock()
	defer a.mu.RUnlock()

	return a.state
}
func (a *App) Wait(ctx context.Context) error {

	if a.State() != Running {
		return ErrInvalidState
	}

	<-ctx.Done()

	return a.Shutdown(context.Background())
}
func New(
	plugins PluginManager,
	events EventBus,
	logger Logger,
	config Config,
	container Container,
	signal SignalWaiter,
) *App {

	app := &App{
		state:   Created,
		plugins: plugins,
		signal:  signal,
	}

	app.ctx = Context{
		App:       app,
		Events:    events,
		Logger:    logger,
		Config:    config,
		Container: container,
	}

	return app
}
func ResolveAs[T any](
	c Container,
	name string,
) (T, error) {

	var zero T

	value, err := c.Resolve(name)

	if err != nil {
		return zero, err
	}

	typed, ok := value.(T)

	if !ok {
		return zero, ErrInvalidDependencyType
	}

	return typed, nil
}

func (c *AudioClock) Advance(
	duration time.Duration,
) {

	c.Position += duration
}

func (f AudioFormat) BytesPerSample() int {
	return f.BitsPerSample / 8
}

func (f AudioFormat) BytesPerSecond() int {
	return f.SampleRate *
		f.Channels *
		f.BytesPerSample()
}

func (f AudioFormat) BytesPerFrame(
	duration time.Duration,
) int {
	return int(
		float64(f.BytesPerSecond()) *
			duration.Seconds(),
	)
}

func (f AudioFrame) Duration() time.Duration {
	if f.Format.BytesPerSecond() == 0 {
		return 0
	}

	seconds := float64(len(f.Data)) /
		float64(f.Format.BytesPerSecond())

	return time.Duration(
		seconds * float64(time.Second),
	)
}
