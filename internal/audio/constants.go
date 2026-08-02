package audio

import (
	"errors"
	"time"
)

const DefaultFrameDuration = 20 * time.Millisecond

var (
	ErrBufferFull  = errors.New("audio ring buffer is full")
	ErrBufferEmpty = errors.New("audio ring buffer is empty")
)
