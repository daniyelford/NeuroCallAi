package llm

import (
	"errors"
)

var (
	ErrInvalidLocalLLM = errors.New("invalid local llm")
	ErrEmptyMessages   = errors.New("empty messages")
	ErrEmptyContent    = errors.New("empty message content")
)
