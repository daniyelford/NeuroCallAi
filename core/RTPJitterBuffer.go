package core

import "time"

func NewRTPJitterBuffer(
	maxPackets int,
	maxWait time.Duration,
) *RTPJitterBuffer {

	if maxPackets <= 0 {
		maxPackets = 50
	}

	if maxWait <= 0 {
		maxWait = 30 * time.Millisecond
	}

	return &RTPJitterBuffer{
		packets:    make(map[uint16]jitterPacket),
		maxPackets: maxPackets,
		maxWait:    maxWait,
		notify:     make(chan struct{}, 1),
	}
}
func (b *RTPJitterBuffer) Push(packet *RTPPacket) error {
	if packet == nil {
		return errInvalidRTPPacket
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	seq := packet.Header.SequenceNumber
	if _, exists := b.packets[seq]; exists {
		return nil
	}
	// Before the buffer has produced anything, allow
	// reordering around the initial sequence number.
	if !b.started {
		b.started = true
		b.next = seq
	}
	// Once output has started, packets behind next are stale.
	if b.advanced && seqLess(seq, b.next) {
		return nil
	}
	b.packets[seq] = jitterPacket{
		packet:     packet,
		receivedAt: time.Now(),
	}
	select {
	case b.notify <- struct{}{}:
	default:
	}
	// Before output starts, an earlier packet becomes
	// the new starting point.
	if !b.advanced && seqLess(seq, b.next) {
		b.next = seq
	}

	if len(b.packets) > b.maxPackets {
		b.dropOldestLocked()
	}

	return nil
}
func (b *RTPJitterBuffer) Pop() (*RTPPacket, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.started {
		return nil, false
	}

	// Expected packet is available.
	if entry, ok := b.packets[b.next]; ok {
		delete(b.packets, b.next)
		b.next++
		b.advanced = true

		return entry.packet, true
	}

	// No packet after the expected sequence.
	if len(b.packets) == 0 {
		return nil, false
	}

	// Something later exists. Check whether the missing
	// packet has waited long enough.
	var oldestWait time.Duration

	for _, entry := range b.packets {
		wait := time.Since(entry.receivedAt)

		if wait > oldestWait {
			oldestWait = wait
		}
	}

	if oldestWait < b.maxWait {
		return nil, false
	}

	// Expected packet is considered lost.
	b.next++
	b.advanced = true

	return nil, true
}
func (b *RTPJitterBuffer) dropOldestLocked() {
	if len(b.packets) == 0 {
		return
	}
	// The oldest packet is the one closest to next
	// in forward RTP sequence space.
	oldest := b.next
	found := false
	var oldestDistance uint16
	for seq := range b.packets {
		distance := seqDistance(b.next, seq)
		if !found || distance < oldestDistance {
			oldest = seq
			oldestDistance = distance
			found = true
		}
	}
	if !found {
		return
	}
	delete(b.packets, oldest)
	// If we dropped the packet we were waiting for,
	// advance next so Pop() can continue.
	if oldest == b.next {
		b.next++
	}
}
func (b *RTPJitterBuffer) WaitForChange(timeout time.Duration) {
	if timeout <= 0 {
		timeout = b.maxWait
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-b.notify:
	case <-timer.C:
	}
}
func (b *RTPJitterBuffer) HasPackets() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.packets) > 0
}
