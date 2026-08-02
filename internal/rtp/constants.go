package rtp

import "errors"

const (
	Version2 = 2

	HeaderSize = 12

	PayloadPCMU uint8 = 0
	PayloadPCMA uint8 = 8
)

var (
	ErrPacketTooShort = errors.New(
		"rtp packet too short",
	)

	ErrInvalidVersion = errors.New(
		"invalid rtp version",
	)

	ErrInvalidPadding = errors.New(
		"invalid rtp padding",
	)
)
