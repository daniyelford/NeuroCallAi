package openaipkg

import (
	"context"
	"fmt"
	"strings"

	openaiSDK "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

type LLM struct {
	client *openaiSDK.Client
	model  string
}

func NewLLM(
	client *openaiSDK.Client,
	model string,
) *LLM {

	if model == "" {
		model = "gpt-6-astra"
	}

	return &LLM{
		client: client,
		model:  model,
	}
}

func (l *LLM) Chat(
	messages []neurocall.Message,
) (neurocall.Message, error) {

	if l == nil || l.client == nil {
		return neurocall.Message{},
			fmt.Errorf("openai llm not configured")
	}

	var parts string

	for _, msg := range messages {
		parts += fmt.Sprintf(
			"%s: %s\n",
			msg.Role,
			msg.Content,
		)
	}

	resp, err := l.client.Responses.New(
		context.Background(),
		responses.ResponseNewParams{

			Model: l.model,

			Input: responses.ResponseNewParamsInputUnion{
				OfString: openaiSDK.String(
					parts,
				),
			},
		},
	)

	if err != nil {
		return neurocall.Message{}, err
	}

	text := strings.TrimSpace(
		resp.OutputText(),
	)

	return neurocall.Message{
		Role:    "assistant",
		Content: text,
	}, nil
}

var _ neurocall.LLM = (*LLM)(nil)
