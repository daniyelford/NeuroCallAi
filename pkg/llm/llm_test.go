package llm

import (
	"testing"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func TestLocalLLMChat(t *testing.T) {
	llm := NewLocalLLM()

	result, err := llm.Chat([]neurocall.Message{
		{
			Role:    "user",
			Content: "hello",
		},
	})

	if err != nil {
		t.Fatal(err)
	}

	if result.Role != "assistant" {
		t.Fatalf("expected assistant role, got %q", result.Role)
	}

	if result.Content == "" {
		t.Fatal("expected non-empty response")
	}
}
