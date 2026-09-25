package llm

import (
	"strings"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewLocalLLM() *LocalLLM {
	return &LocalLLM{}
}

func (l *LocalLLM) Chat(
	messages []neurocall.Message,
) (neurocall.Message, error) {

	if l == nil {
		return neurocall.Message{}, ErrInvalidLocalLLM
	}

	if len(messages) == 0 {
		return neurocall.Message{}, ErrEmptyMessages
	}

	var lastUserMessage string

	for i := len(messages) - 1; i >= 0; i-- {
		message := messages[i]

		if strings.TrimSpace(message.Content) == "" {
			continue
		}

		if strings.EqualFold(message.Role, "user") {
			lastUserMessage = strings.TrimSpace(message.Content)
			break
		}
	}

	if lastUserMessage == "" {
		return neurocall.Message{}, ErrEmptyContent
	}

	return neurocall.Message{
		Role:    "assistant",
		Content: localResponse(lastUserMessage),
	}, nil
}

func localResponse(input string) string {
	input = strings.TrimSpace(input)

	switch strings.ToLower(input) {
	case "hello":
		return "Hello. How can I help you?"

	case "hi":
		return "Hello. How can I help you?"

	case "how are you":
		return "I'm fine. How can I help you?"

	default:
		return "I received your message: " + input
	}
}
