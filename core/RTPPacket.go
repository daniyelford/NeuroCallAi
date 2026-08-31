package core

import (
	"encoding/binary"
	"fmt"
)

func (p *RTPPacket) Marshal() ([]byte, error) {
	if p == nil {
		return nil, errInvalidRTPPacket
	}

	if p.Header.Version != 2 {
		return nil, errInvalidRTPPacket
	}

	if p.Header.CSRCCount != 0 {
		return nil, fmt.Errorf(
			"CSRC is not supported yet",
		)
	}

	data := make(
		[]byte,
		12+len(p.Payload),
	)

	data[0] = p.Header.Version << 6

	if p.Header.Padding {
		data[0] |= 1 << 5
	}

	if p.Header.Extension {
		data[0] |= 1 << 4
	}

	data[0] |= p.Header.CSRCCount & 0x0F

	data[1] = p.Header.PayloadType & 0x7F

	if p.Header.Marker {
		data[1] |= 1 << 7
	}

	binary.BigEndian.PutUint16(
		data[2:4],
		p.Header.SequenceNumber,
	)

	binary.BigEndian.PutUint32(
		data[4:8],
		p.Header.Timestamp,
	)

	binary.BigEndian.PutUint32(
		data[8:12],
		p.Header.SSRC,
	)

	copy(
		data[12:],
		p.Payload,
	)

	return data, nil
}
func (p *RTPPacket) Unmarshal(
	data []byte,
) error {
	if len(data) < 12 {
		return errInvalidRTPPacket
	}
	version := data[0] >> 6
	if version != 2 {
		return errInvalidRTPPacket
	}
	padding := data[0]&0x20 != 0
	extension := data[0]&0x10 != 0
	csrcCount := data[0] & 0x0F
	if csrcCount != 0 {
		return fmt.Errorf("CSRC is not supported yet")
	}
	header := RTPHeader{
		Version:        version,
		Padding:        padding,
		Extension:      extension,
		CSRCCount:      csrcCount,
		Marker:         data[1]&0x80 != 0,
		PayloadType:    data[1] & 0x7F,
		SequenceNumber: binary.BigEndian.Uint16(data[2:4]),
		Timestamp:      binary.BigEndian.Uint32(data[4:8]),
		SSRC:           binary.BigEndian.Uint32(data[8:12]),
	}
	offset := 12
	if extension {
		if len(data) < offset+4 {
			return errInvalidRTPPacket
		}
		extensionLength := int(binary.BigEndian.Uint16(data[offset+2:offset+4])) * 4
		offset += 4
		if len(data) < offset+extensionLength {
			return errInvalidRTPPacket
		}
		offset += extensionLength
	}
	payload := data[offset:]
	if padding {
		if len(payload) == 0 {
			return errInvalidRTPPacket
		}
		paddingLen := int(payload[len(payload)-1])
		if paddingLen == 0 || paddingLen > len(payload) {
			return errInvalidRTPPacket
		}
		payload = payload[:len(payload)-paddingLen]
	}
	p.Payload = append(
		[]byte(nil),
		payload...,
	)
	p.Header = header
	return nil
}
