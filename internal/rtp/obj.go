package rtp

import (
	"net"
	"sync"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

type Receiver struct {
	conn *net.UDPConn
}
type JitterBuffer struct {
	mu sync.Mutex

	packets map[uint16]neurocall.RTPPacket

	next uint16

	started bool

	capacity int
}
