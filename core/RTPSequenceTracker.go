package core

func NewRTPSequenceTracker() *RTPSequenceTracker {
	return &RTPSequenceTracker{}
}
func (t *RTPSequenceTracker) Update(seq uint16) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.started {
		t.started = true
		t.last = seq
		t.received++
		return
	}

	// Same sequence = duplicate.
	if seq == t.last {
		t.duplicate++
		return
	}

	diff := uint16(seq - t.last)

	// Forward packet.
	//
	// This also correctly handles:
	//
	// 65534 -> 65535
	// 65535 -> 0
	// 0 -> 1
	//
	if diff < 0x8000 {
		if diff > 1 {
			t.lost += uint64(diff - 1)
		}

		t.last = seq
		t.received++
		return
	}

	// Packet arrived behind current sequence.
	t.outOfOrder++
}
func (t *RTPSequenceTracker) Stats() RTPStats {
	t.mu.Lock()
	defer t.mu.Unlock()

	return RTPStats{
		ReceivedPackets: t.received,
		LostPackets:     t.lost,
		Duplicates:      t.duplicate,
		OutOfOrder:      t.outOfOrder,

		LastSequence: t.last,
		HasSequence:  t.started,
	}
}
func (t *RTPSequenceTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.started = false
	t.last = 0
	t.received = 0
	t.lost = 0
	t.duplicate = 0
	t.outOfOrder = 0
}
