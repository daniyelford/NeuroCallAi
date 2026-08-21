package neurocall

import "errors"

var (
	ErrNoCommonCodec = errors.New(
		"no common audio codec",
	)

	ErrCallNotFound = errors.New(
		"call not found",
	)

	ErrCallClosed = errors.New(
		"call is closed",
	)

	ErrToolNotFound = errors.New(
		"tool not found",
	)

	ErrInvalidAudio = errors.New(
		"invalid audio",
	)
)

const (
	STTPartial STTEventType = iota
	STTFinal
	STTError
	// roles
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
	// events
	EventAIStarted     = "ai.started"
	EventAIThinking    = "ai.thinking"
	EventAIResponse    = "ai.response"
	EventAIToolCall    = "ai.tool_call"
	EventAIToolResult  = "ai.tool_result"
	EventAIInterrupted = "ai.interrupted"
	EventAIError       = "ai.error"
)
