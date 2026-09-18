package openaipkg

import (
	"bytes"
	"encoding/binary"
)

func PCM16ToWAV(
	pcm []int16,
	sampleRate int,
	channels int,
) []byte {

	var buf bytes.Buffer

	dataSize := len(pcm) * 2

	buf.Write([]byte("RIFF"))

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint32(36+dataSize),
	)

	buf.Write([]byte("WAVE"))

	buf.Write([]byte("fmt "))

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint32(16),
	)

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint16(1),
	)

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint16(channels),
	)

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint32(sampleRate),
	)

	byteRate :=
		sampleRate *
			channels *
			2

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint32(byteRate),
	)

	blockAlign :=
		uint16(channels * 2)

	binary.Write(
		&buf,
		binary.LittleEndian,
		blockAlign,
	)

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint16(16),
	)

	buf.Write([]byte("data"))

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint32(dataSize),
	)

	for _, sample := range pcm {
		binary.Write(
			&buf,
			binary.LittleEndian,
			sample,
		)
	}

	return buf.Bytes()
}
