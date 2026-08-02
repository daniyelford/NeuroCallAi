package neurocall

import (
	"context"
	"sync"
	"time"
)

type Context struct {
	App       *App
	Events    EventBus
	Logger    Logger
	Config    Config
	Container Container
}

// import "github.com/daniyelford/neurocall/internal/plugin"
type EventHandler func(Event)

//	type Container interface {
//		Register(name string, value any) error
//		Resolve(name string) (any, error)
//	}
type EventBus interface {
	Publish(Event)
	Subscribe(string, EventHandler) Subscription
}
type Manager struct {
	registry *Registry
}
type AudioClock struct {
	Position time.Duration
}
type Event interface {
	Name() string
	Time() time.Time
}
type BasicEvent struct {
	EventName string
	CreatedAt time.Time
}

//	type Context struct {
//		App *App
//	}
type Logger interface {
	Debug(msg string, args ...any)
	Info(msg string, args ...any)
	Warn(msg string, args ...any)
	Error(msg string, args ...any)
}
type Config interface {
	Get(key string) (string, bool)
}
type App struct {
	mu sync.RWMutex

	state State

	ctx Context

	plugins PluginManager
	signal  SignalWaiter
}

//	type Logger interface {
//		Debug(msg string)
//		Info(msg string)
//		Warn(msg string)
//		Error(msg string)
//	}
type Plugin interface {
	Name() string
	DependsOn() []string
	Init(*Context) error
	Start() error
	Stop() error
}
type State uint8

//	type App struct {
//		state State
//		ctx   Context
//		// plugins []Plugin
//		registry *Registry
//	}
type Option func(*App)

type Subscription interface {
	Unsubscribe()
}

type PluginManager interface {
	Register(Plugin) error

	Init(*Context) error

	Start() error

	Stop() error
}
type SignalWaiter interface {
	Wait(Context) error
}
type Container interface {
	Register(name string, value any) error

	Resolve(name string) (any, error)
}

//	type Context struct {
//		App    *App
//		Events EventBus
//	}

//	type AudioFrame struct {
//		Data      []byte
//		Format    AudioFormat
//		Timestamp time.Time
//	}
type AudioSource interface {
	Read(ctx context.Context) (AudioFrame, error)
}
type AudioSink interface {
	Write(
		ctx context.Context,
		frame AudioFrame,
	) error
}
type AudioProcessor interface {
	Process(
		context.Context,
		AudioFrame,
	) (AudioFrame, error)
}
type AudioFormat struct {
	SampleRate    int
	Channels      int
	BitsPerSample int
}
type AudioFrame struct {
	Data      []byte
	Format    AudioFormat
	Timestamp time.Duration
}
type SpeechSegment struct {
	Frames []AudioFrame

	Start time.Duration
	End   time.Duration
}
