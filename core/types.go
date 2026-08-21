package core

import (
	"context"
	"log"
	"net"
	"sync"
	"time"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

// --------------------
// SIP
// --------------------

type SIPMessage struct {
	StartLine string
	Headers   map[string]string
	Body      []byte
}
type SIPConfig struct {
	ListenIP   string
	SIPPort    int
	RTPMinPort int
	RTPMaxPort int
	ExternalIP string
	Domain     string
}
type AIConfig struct {
	STT neurocall.STT
	LLM neurocall.LLM
	TTS neurocall.TTS
}
type SIPServer struct {
	mu           sync.RWMutex
	conn         *net.UDPConn
	manager      *CallManager
	bus          *EventBus
	codecs       *CodecRegistry
	config       SIPConfig
	stt          *STTEngine
	llm          *LLMEngine
	tts          *TTSEngine
	conversation *ConversationEngine
	voice        *VoiceResponseEngine
	ctx          context.Context
	cancel       context.CancelFunc
}
type SIPDialog struct {
	CallID    string
	LocalTag  string
	RemoteTag string
	LocalURI  string
	RemoteURI string
	LocalSeq  uint32
	RemoteSeq uint32
}
type RemoteMedia struct {
	IP     net.IP
	Port   int
	Codecs []neurocall.Codec
}
type CallSession struct {
	mu           sync.RWMutex
	ID           string
	Call         *SIPCall
	Audio        *AudioPipeline
	STT          neurocall.STT
	StreamingSTT neurocall.StreamingSTT
	STTWorker    *STTWorker
	TTS          neurocall.TTS
	LLM          neurocall.LLM
	Memory       neurocall.Memory
	Tools        neurocall.ToolRegistry
	Events       *EventBus
	ctx          context.Context
	cancel       context.CancelFunc
	running      bool
	closed       bool
}
type CallManager struct {
	mu       sync.RWMutex
	calls    map[string]*SIPCall
	ports    *PortAllocator
	sessions map[string]*CallSession
}

// --------------------
// RTP
// --------------------

type PortAllocator struct {
	mu        sync.Mutex
	min       int
	max       int
	available map[int]struct{}
}
type CodecRegistry struct {
	preferred []neurocall.Codec
}

// --------------------
// Audio
// --------------------

type AudioConfig struct {
	SampleRate int
	Channels   int
	FrameSize  int
	BufferSize int
}
type AudioBuffer struct {
	mu       sync.Mutex
	data     []int16
	capacity int
}
type AudioPipeline struct {
	mu        sync.RWMutex
	encoder   neurocall.Encoder
	decoder   neurocall.Decoder
	resampler neurocall.Resampler
	input     chan []int16
	output    chan []int16
	ctx       context.Context
	cancel    context.CancelFunc
}
type AudioEngine struct {
	pipeline *AudioPipeline
	running  bool
	mu       sync.RWMutex
}
type PassthroughAudio struct{}
type Pipeline struct {
	codec     neurocall.Codec
	rtp       *RTPSession
	encoder   neurocall.Encoder
	decoder   neurocall.Decoder
	resampler neurocall.Resampler
}

// --------------------
// Event
// --------------------

type Event struct {
	Name string
	Data any
}
type EventHandler func(Event)
type EventBus struct {
	mu       sync.RWMutex
	handlers map[string][]EventHandler
}

// --------------------
// Shutdown
// --------------------

type ShutdownManager struct {
	mu           sync.Mutex
	handlers     []func() error
	shuttingDown bool
}

// --------------------
// Logger
// --------------------

type Logger struct {
	logger *log.Logger
}

// --------------------
// Config
// --------------------

type Config struct {
	SIP        SIPConfig
	Audio      AudioConfig
	ExternalIP string
	Domain     string
	RTPMinPort int
	RTPMaxPort int
	AI         AIConfig
}

// --------------------
// Container
// --------------------
type Container struct {
	mu       sync.RWMutex
	services map[string]any
}

// --------------------
// Plugin
// --------------------
type Plugin interface {
	Name() string
	Init(*Container) error
	Start(context.Context) error
	Stop(context.Context) error
}
type PluginRegistry struct {
	mu      sync.RWMutex
	plugins map[string]Plugin
}
type PluginGraph struct {
	mu           sync.RWMutex
	dependencies map[string][]string
}

// --------------------
// Codec
// --------------------
type PCMU struct{}
type PCMA struct{}
type MapMemory struct {
	mu   sync.RWMutex
	data map[string]any
}
type ToolRegistry struct {
	mu    sync.RWMutex
	tools map[string]neurocall.Tool
}
type STTEngine struct {
	mu     sync.RWMutex
	stt    neurocall.STT
	ctx    context.Context
	cancel context.CancelFunc
}
type SIPCall struct {
	CallID       string
	TTS          *TTSPlayback
	RemoteIP     net.IP
	RemotePort   int
	LocalRTPPort int
	Codec        neurocall.Codec
	RemoteAddr   *net.UDPAddr
	RTP          *RTPSession
	Pipeline     *AudioPipeline
	VAD          neurocall.VAD
	Segmenter    neurocall.SpeechSegmenter
	STT          *STTWorker
	Conversation *Conversation
	dialog       *SIPDialog
	answered     bool
	closed       bool
	mu           sync.RWMutex
}
type EnergyVAD struct {
	Threshold int64
}
type AudioSegmenter struct {
	mu            sync.Mutex
	vad           neurocall.VAD
	buffer        []int16
	active        bool
	start         time.Duration
	end           time.Duration
	silenceFrames int
	maxFrames     int
	frameDuration time.Duration
	sampleRate    int
	channels      int
	frameCount    int
}
type TranscriptEvent struct {
	CallID     string
	Transcript neurocall.Transcript
}
type STTWorker struct {
	engine    *STTEngine
	bus       *EventBus
	callID    string
	input     chan neurocall.AudioSegment
	streaming neurocall.StreamingSTT
	ctx       context.Context
	cancel    context.CancelFunc
	mu        sync.Mutex
	running   bool
	stopped   bool
}
type STTErrorEvent struct {
	CallID string
	Err    error
}
type LLMEngine struct {
	mu  sync.RWMutex
	llm neurocall.LLM
}
type CallMemory struct {
	mu     sync.RWMutex
	values map[string]any
}
type Conversation struct {
	mu       sync.RWMutex
	messages []neurocall.Message
	memory   *CallMemory
}
type ConversationEngine struct {
	mu            sync.RWMutex
	llm           *LLMEngine
	conversations map[string]*Conversation
	bus           *EventBus
}

type LLMResponseEvent struct {
	CallID  string
	Message neurocall.Message
}
type TTSEngine struct {
	mu  sync.RWMutex
	tts neurocall.TTS
}
type TTSPlayback struct {
	engine *TTSEngine
	ctx    context.Context
	cancel context.CancelFunc
}
type VoiceResponseEngine struct {
	mu    sync.RWMutex
	calls map[string]*SIPCall
	tts   *TTSEngine
	bus   *EventBus
}
type RTPHeader struct {
	Version        uint8
	Padding        bool
	Extension      bool
	CSRCCount      uint8
	Marker         bool
	PayloadType    uint8
	SequenceNumber uint16
	Timestamp      uint32
	SSRC           uint32
}
type RTPPacket struct {
	Header  RTPHeader
	Payload []byte
}
type RTPSession struct {
	mu              sync.RWMutex
	conn            *net.UDPConn
	remote          *net.UDPAddr
	codec           neurocall.Codec
	ssrc            uint32
	sequence        uint16
	timestamp       uint32
	closed          bool
	sequenceTracker *RTPSequenceTracker
	jitter          *RTPJitterBuffer
	stats           RTPStats
	writeInterval   time.Duration
}
type RTPStats struct {
	ReceivedPackets uint64
	LostPackets     uint64
	InvalidPackets  uint64
	SentPackets     uint64
	BytesReceived   uint64
	BytesSent       uint64
	// DroppedPackets  uint64
	Duplicates   uint64
	OutOfOrder   uint64
	LastSequence uint16
	HasSequence  bool
}
type RTPSequenceTracker struct {
	mu         sync.Mutex
	last       uint16
	started    bool
	received   uint64
	lost       uint64
	duplicate  uint64
	outOfOrder uint64
}
type jitterPacket struct {
	packet     *RTPPacket
	receivedAt time.Time
}
type RTPJitterBuffer struct {
	mu         sync.Mutex
	packets    map[uint16]jitterPacket
	next       uint16
	started    bool
	maxPackets int
	maxWait    time.Duration
}
type RTPReadResult struct {
	Packet *RTPPacket
	Lost   bool
}
type STTEventData struct {
	CallID string
	Event  neurocall.STTEvent
}
type TTSErrorEvent struct {
	CallID string
	Err    error
}
type FakeSTT struct{}

//	type TTSJob struct {
//		CallID string
//		Text   string
//	}
//
//	type TTSPlaybackQueue struct {
//		mu     sync.Mutex
//		jobs   chan TTSJob
//		ctx    context.Context
//		cancel context.CancelFunc
//	}
