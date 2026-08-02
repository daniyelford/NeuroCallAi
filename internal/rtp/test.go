package rtp

import (
	"testing"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

func TestEncodeDecodeRoundTrip(t *testing.T) {

	original := neurocall.RTPPacket{
		Version:        2,
		PayloadType:    PayloadPCMU,
		SequenceNumber: 100,
		Timestamp:      1600,
		SSRC:           12345,
		Payload:        []byte{1, 2, 3, 4},
	}

	data, err := Encode(original)

	if err != nil {
		t.Fatal(err)
	}

	decoded, err := Decode(data)

	if err != nil {
		t.Fatal(err)
	}

	if decoded.SequenceNumber !=
		original.SequenceNumber {
		t.Fatal("sequence mismatch")
	}

	if decoded.Timestamp !=
		original.Timestamp {
		t.Fatal("timestamp mismatch")
	}

	if decoded.SSRC !=
		original.SSRC {
		t.Fatal("ssrc mismatch")
	}
}
