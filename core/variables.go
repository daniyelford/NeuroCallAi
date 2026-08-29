package core

import (
	"errors"
	"log"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

const (
	// SIP
	DefaultSIPPort = 5060
	// RTP
	DefaultRTPMinPort = 10000
	DefaultRTPMaxPort = 20000
	// Audio
	DefaultAudioSampleRate = 8000
	DefaultAudioChannels   = 1
	DefaultAudioFrameSize  = 160
	// Events
	EventAudioSegment        = "audio.segment"
	EventAudioProcessorError = "audio.processor.error"
	EventTTSResponseError    = "tts.response.error"
	EventLLMError            = "llm.error"
	EventSTTEvent            = "stt.event"
	EventSTTError            = "stt.error"
	EventCallStarted         = "call.started"
	EventCallAnswered        = "call.answered"
	EventCallEnded           = "call.ended"
	EventAudioReceived       = "audio.received"
	EventAudioSent           = "audio.sent"
	EventTranscript          = "audio.transcript"
	EventToolCall            = "tool.call"
	EventError               = "error"
	EventLLMResponse         = "llm.response"
)

var (
	// rtp
	errInvalidRTPPacket = errors.New(
		"invalid RTP packet",
	)
	errRTPSessionClosed = errors.New(
		"RTP session is closed",
	)
	// sip
	errInvalidSIPMessage = errors.New(
		"invalid SIP message",
	)
	errNoAudioMedia = errors.New(
		"no audio media",
	)
	errNoAudioCodec = errors.New(
		"no audio codecs",
	)
	errRTPPortExhausted = errors.New(
		"RTP port range exhausted",
	)
	defaultCodecRegistry = []neurocall.Codec{
		{
			Name:        "PCMU",
			PayloadType: 0,
			ClockRate:   8000,
			Channels:    1,
		},
		{
			Name:        "PCMA",
			PayloadType: 8,
			ClockRate:   8000,
			Channels:    1,
		},
	}
	defaultLogger = log.Default()
)
var _ neurocall.Call = (*SIPCall)(nil)
var _ neurocall.ToolRegistry = (*ToolRegistry)(nil)
