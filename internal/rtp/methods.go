package rtp

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"time"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

func Decode(
	data []byte,
) (neurocall.RTPPacket, error) {

	if len(data) < HeaderSize {
		return neurocall.RTPPacket{},
			ErrPacketTooShort
	}

	packet := neurocall.RTPPacket{}

	first := data[0]
	second := data[1]

	packet.Version = first >> 6

	if packet.Version != Version2 {
		return neurocall.RTPPacket{},
			fmt.Errorf(
				"%w: %d",
				ErrInvalidVersion,
				packet.Version,
			)
	}

	packet.Padding = first&0x20 != 0
	packet.Extension = first&0x10 != 0
	packet.CSRCCount = first & 0x0F

	packet.Marker = second&0x80 != 0
	packet.PayloadType = second & 0x7F

	packet.SequenceNumber =
		binary.BigEndian.Uint16(data[2:4])

	packet.Timestamp =
		binary.BigEndian.Uint32(data[4:8])

	packet.SSRC =
		binary.BigEndian.Uint32(data[8:12])

	offset := HeaderSize

	csrcBytes := int(packet.CSRCCount) * 4

	if len(data) < offset+csrcBytes {
		return neurocall.RTPPacket{},
			ErrPacketTooShort
	}

	packet.CSRC = make(
		[]uint32,
		packet.CSRCCount,
	)

	for i := 0; i < int(packet.CSRCCount); i++ {

		packet.CSRC[i] =
			binary.BigEndian.Uint32(
				data[offset+i*4:],
			)
	}

	offset += csrcBytes

	if packet.Extension {

		if len(data) < offset+4 {
			return neurocall.RTPPacket{},
				ErrPacketTooShort
		}

		extensionLength :=
			int(binary.BigEndian.Uint16(
				data[offset+2:offset+4],
			)) * 4

		offset += 4

		if len(data) < offset+extensionLength {
			return neurocall.RTPPacket{},
				ErrPacketTooShort
		}

		offset += extensionLength
	}

	if offset > len(data) {
		return neurocall.RTPPacket{},
			ErrPacketTooShort
	}

	payloadEnd := len(data)

	if packet.Padding {

		padding := int(data[len(data)-1])

		if padding == 0 || padding > payloadEnd-offset {
			return neurocall.RTPPacket{},
				ErrInvalidPadding
		}

		payloadEnd -= padding
	}

	packet.Payload = append(
		[]byte(nil),
		data[offset:payloadEnd]...,
	)

	return packet, nil
}
func Encode(
	packet neurocall.RTPPacket,
) ([]byte, error) {

	if packet.Version == 0 {
		packet.Version = Version2
	}

	if packet.Version != Version2 {
		return nil, fmt.Errorf(
			"%w: %d",
			ErrInvalidVersion,
			packet.Version,
		)
	}

	if len(packet.CSRC) > 15 {
		return nil, errors.New(
			"too many csrc identifiers",
		)
	}

	headerSize :=
		HeaderSize +
			len(packet.CSRC)*4

	output := make(
		[]byte,
		headerSize+len(packet.Payload),
	)

	output[0] =
		(packet.Version << 6) |
			uint8(len(packet.CSRC))

	if packet.Padding {
		output[0] |= 0x20
	}

	if packet.Extension {
		output[0] |= 0x10
	}

	output[1] = packet.PayloadType & 0x7F

	if packet.Marker {
		output[1] |= 0x80
	}

	binary.BigEndian.PutUint16(
		output[2:4],
		packet.SequenceNumber,
	)

	binary.BigEndian.PutUint32(
		output[4:8],
		packet.Timestamp,
	)

	binary.BigEndian.PutUint32(
		output[8:12],
		packet.SSRC,
	)

	offset := HeaderSize

	for _, csrc := range packet.CSRC {

		binary.BigEndian.PutUint32(
			output[offset:offset+4],
			csrc,
		)

		offset += 4
	}

	copy(
		output[offset:],
		packet.Payload,
	)

	return output, nil
}
func MuLawDecode(v byte) int16 {

	v = ^v

	sign := v & 0x80
	exponent := (v >> 4) & 0x07
	mantissa := v & 0x0F

	sample :=
		((int(mantissa) << 3) + 0x84) <<
			exponent

	if sign != 0 {
		return int16(0x84 - sample)
	}

	return int16(sample - 0x84)
}
func ALawDecode(v byte) int16 {

	v ^= 0x55

	sign := v & 0x80

	exponent := (v >> 4) & 0x07

	mantissa := v & 0x0F

	var sample int

	if exponent == 0 {

		sample =
			(int(mantissa) << 4) +
				8

	} else {

		sample =
			((int(mantissa) << 4) +
				0x108) <<
				(exponent - 1)
	}

	if sign != 0 {
		return int16(sample)
	}

	return int16(-sample)
}

func MuLawToPCM16(
	input []byte,
) []byte {

	output := make(
		[]byte,
		len(input)*2,
	)

	for i, value := range input {

		sample := MuLawDecode(value)

		binary.LittleEndian.PutUint16(
			output[i*2:],
			uint16(sample),
		)
	}

	return output
}

func ALawToPCM16(
	input []byte,
) []byte {

	output := make(
		[]byte,
		len(input)*2,
	)

	for i, value := range input {

		sample := ALawDecode(value)

		binary.LittleEndian.PutUint16(
			output[i*2:],
			uint16(sample),
		)
	}

	return output
}
func ToAudioFrame(
	packet neurocall.RTPPacket,
) (neurocall.AudioFrame, error) {

	var pcm []byte

	var format neurocall.AudioFormat

	switch packet.PayloadType {

	case PayloadPCMU:

		pcm = MuLawToPCM16(
			packet.Payload,
		)

		format = neurocall.AudioFormat{
			SampleRate:    8000,
			Channels:      1,
			BitsPerSample: 16,
		}

	case PayloadPCMA:

		pcm = ALawToPCM16(
			packet.Payload,
		)

		format = neurocall.AudioFormat{
			SampleRate:    8000,
			Channels:      1,
			BitsPerSample: 16,
		}

	default:

		return neurocall.AudioFrame{},
			fmt.Errorf(
				"unsupported RTP payload type: %d",
				packet.PayloadType,
			)
	}

	return neurocall.AudioFrame{
		Data:   pcm,
		Format: format,
		Timestamp: time.Duration(packet.Timestamp) *
			time.Second /
			time.Duration(format.SampleRate),
	}, nil
}
func NewReceiver(
	conn *net.UDPConn,
) *Receiver {
	return &Receiver{
		conn: conn,
	}
}

func (r *Receiver) Read(
	ctx context.Context,
) (neurocall.AudioFrame, error) {

	buffer := make([]byte, 2048)

	for {

		if err := r.conn.SetReadDeadline(
			nextDeadline(ctx),
		); err != nil {
			return neurocall.AudioFrame{}, err
		}

		n, _, err := r.conn.ReadFromUDP(buffer)

		if err != nil {

			if ctx.Err() != nil {
				return neurocall.AudioFrame{},
					ctx.Err()
			}

			if netErr, ok := err.(net.Error); ok &&
				netErr.Timeout() {
				continue
			}

			return neurocall.AudioFrame{}, err
		}

		packet, err := Decode(buffer[:n])

		if err != nil {
			continue
		}

		return ToAudioFrame(packet)
	}
}
func nextDeadline(ctx context.Context) time.Time {

	deadline := time.Now().Add(500 * time.Millisecond)

	if d, ok := ctx.Deadline(); ok &&
		d.Before(deadline) {
		return d
	}

	return deadline
}
