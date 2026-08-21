package neurocall

import (
	"errors"
	"testing"
	"time"
)

func TestNewCodec(t *testing.T) {
	codec := NewCodec(
		"PCMU",
		0,
		8000,
		1,
	)

	if codec.Name != "PCMU" {
		t.Errorf("expected name PCMU, got %q", codec.Name)
	}

	if codec.PayloadType != 0 {
		t.Errorf("expected payload type 0, got %d", codec.PayloadType)
	}

	if codec.ClockRate != 8000 {
		t.Errorf("expected clock rate 8000, got %d", codec.ClockRate)
	}

	if codec.Channels != 1 {
		t.Errorf("expected channels 1, got %d", codec.Channels)
	}

	if !codec.Valid() {
		t.Error("expected codec to be valid")
	}
}
func TestCodecValid(t *testing.T) {
	tests := []struct {
		name  string
		codec Codec
		valid bool
	}{
		{
			name: "valid codec",
			codec: Codec{
				Name:        "PCMU",
				PayloadType: 0,
				ClockRate:   8000,
				Channels:    1,
			},
			valid: true,
		},
		{
			name: "missing name",
			codec: Codec{
				ClockRate: 8000,
				Channels:  1,
			},
			valid: false,
		},
		{
			name: "invalid clock rate",
			codec: Codec{
				Name:     "PCMU",
				Channels: 1,
			},
			valid: false,
		},
		{
			name: "invalid channels",
			codec: Codec{
				Name:      "PCMU",
				ClockRate: 8000,
			},
			valid: false,
		},
		{
			name:  "all invalid",
			codec: Codec{},
			valid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.codec.Valid()

			if got != tt.valid {
				t.Errorf(
					"Valid() = %v, expected %v",
					got,
					tt.valid,
				)
			}
		})
	}
}
func TestMessageValid(t *testing.T) {
	tests := []struct {
		name    string
		message Message
		valid   bool
	}{
		{
			name: "role and content",
			message: Message{
				Role:    RoleUser,
				Content: "hello",
			},
			valid: true,
		},
		{
			name: "role only",
			message: Message{
				Role: RoleUser,
			},
			valid: true,
		},
		{
			name: "content only",
			message: Message{
				Content: "hello",
			},
			valid: true,
		},
		{
			name:    "empty",
			message: Message{},
			valid:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.message.Valid()

			if got != tt.valid {
				t.Errorf(
					"Valid() = %v, expected %v",
					got,
					tt.valid,
				)
			}
		})
	}
}
func TestErrorValues(t *testing.T) {
	if ErrNoCommonCodec == nil {
		t.Error("ErrNoCommonCodec must not be nil")
	}

	if ErrCallNotFound == nil {
		t.Error("ErrCallNotFound must not be nil")
	}

	if ErrCallClosed == nil {
		t.Error("ErrCallClosed must not be nil")
	}

	if ErrToolNotFound == nil {
		t.Error("ErrToolNotFound must not be nil")
	}

	if ErrInvalidAudio == nil {
		t.Error("ErrInvalidAudio must not be nil")
	}
}
func TestErrorIdentity(t *testing.T) {
	if !errors.Is(ErrNoCommonCodec, ErrNoCommonCodec) {
		t.Error("ErrNoCommonCodec should identify itself")
	}

	if !errors.Is(ErrCallNotFound, ErrCallNotFound) {
		t.Error("ErrCallNotFound should identify itself")
	}

	if !errors.Is(ErrCallClosed, ErrCallClosed) {
		t.Error("ErrCallClosed should identify itself")
	}

	if !errors.Is(ErrToolNotFound, ErrToolNotFound) {
		t.Error("ErrToolNotFound should identify itself")
	}

	if !errors.Is(ErrInvalidAudio, ErrInvalidAudio) {
		t.Error("ErrInvalidAudio should identify itself")
	}
}
func TestConstants(t *testing.T) {
	if RoleSystem != "system" {
		t.Errorf("unexpected RoleSystem: %q", RoleSystem)
	}

	if RoleUser != "user" {
		t.Errorf("unexpected RoleUser: %q", RoleUser)
	}

	if RoleAssistant != "assistant" {
		t.Errorf("unexpected RoleAssistant: %q", RoleAssistant)
	}

	if RoleTool != "tool" {
		t.Errorf("unexpected RoleTool: %q", RoleTool)
	}

	if EventAIStarted != "ai.started" {
		t.Errorf("unexpected EventAIStarted: %q", EventAIStarted)
	}

	if EventAIThinking != "ai.thinking" {
		t.Errorf("unexpected EventAIThinking: %q", EventAIThinking)
	}

	if EventAIResponse != "ai.response" {
		t.Errorf("unexpected EventAIResponse: %q", EventAIResponse)
	}

	if EventAIToolCall != "ai.tool_call" {
		t.Errorf("unexpected EventAIToolCall: %q", EventAIToolCall)
	}

	if EventAIToolResult != "ai.tool_result" {
		t.Errorf("unexpected EventAIToolResult: %q", EventAIToolResult)
	}

	if EventAIInterrupted != "ai.interrupted" {
		t.Errorf("unexpected EventAIInterrupted: %q", EventAIInterrupted)
	}

	if EventAIError != "ai.error" {
		t.Errorf("unexpected EventAIError: %q", EventAIError)
	}
}
func TestSTTEventTypes(t *testing.T) {
	if STTPartial != STTEventType(0) {
		t.Errorf("STTPartial = %d, expected 0", STTPartial)
	}

	if STTFinal != STTEventType(1) {
		t.Errorf("STTFinal = %d, expected 1", STTFinal)
	}

	if STTError != STTEventType(2) {
		t.Errorf("STTError = %d, expected 2", STTError)
	}
}
func TestAudioStructures(t *testing.T) {
	frame := AudioFrame{
		Data:       []int16{1, 2, 3, 4},
		Timestamp:  100,
		SampleRate: 8000,
		Channels:   1,
	}

	if len(frame.Data) != 4 {
		t.Errorf("expected 4 samples, got %d", len(frame.Data))
	}

	if frame.Timestamp != 100 {
		t.Errorf("expected timestamp 100, got %d", frame.Timestamp)
	}

	if frame.SampleRate != 8000 {
		t.Errorf("expected sample rate 8000, got %d", frame.SampleRate)
	}

	if frame.Channels != 1 {
		t.Errorf("expected 1 channel, got %d", frame.Channels)
	}
}
func TestAudioSegment(t *testing.T) {
	segment := AudioSegment{
		Data:       []int16{10, 20, 30},
		SampleRate: 16000,
		Channels:   1,
		Start:      100 * time.Millisecond,
		End:        300 * time.Millisecond,
		Final:      true,
	}

	if len(segment.Data) != 3 {
		t.Errorf("expected 3 samples, got %d", len(segment.Data))
	}

	if segment.SampleRate != 16000 {
		t.Errorf("expected sample rate 16000, got %d", segment.SampleRate)
	}

	if segment.Start != 100*time.Millisecond {
		t.Errorf("unexpected start time: %v", segment.Start)
	}

	if segment.End != 300*time.Millisecond {
		t.Errorf("unexpected end time: %v", segment.End)
	}

	if !segment.Final {
		t.Error("expected segment to be final")
	}
}
func TestAudioFormat(t *testing.T) {
	format := AudioFormat{
		SampleRate: 8000,
		Channels:   1,
		FrameSize:  160,
		Codec:      "PCMU",
	}

	if format.SampleRate != 8000 {
		t.Errorf("unexpected sample rate: %d", format.SampleRate)
	}

	if format.Channels != 1 {
		t.Errorf("unexpected channels: %d", format.Channels)
	}

	if format.FrameSize != 160 {
		t.Errorf("unexpected frame size: %d", format.FrameSize)
	}

	if format.Codec != "PCMU" {
		t.Errorf("unexpected codec: %q", format.Codec)
	}
}
