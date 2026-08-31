package core

import (
	"fmt"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewLLMEngine(
	llm neurocall.LLM,
) *LLMEngine {
	return &LLMEngine{
		llm: llm,
	}
}
func (e *LLMEngine) SetLLM(
	llm neurocall.LLM,
) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.llm = llm
}
func (e *LLMEngine) Chat(
	messages []neurocall.Message,
) (neurocall.Message, error) {
	e.mu.RLock()
	llm := e.llm
	e.mu.RUnlock()
	if llm == nil {
		return neurocall.Message{},
			fmt.Errorf(
				"LLM is not configured",
			)
	}
	return llm.Chat(messages)
}
