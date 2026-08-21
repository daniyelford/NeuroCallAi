package core

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
	"github.com/pion/sdp/v3"
)

func PCM16FromBytes(
	data []byte,
) []int16 {

	if len(data)%2 != 0 {
		data = data[:len(data)-1]
	}

	out := make(
		[]int16,
		len(data)/2,
	)

	for i := range out {
		out[i] =
			int16(data[i*2]) |
				int16(data[i*2+1])<<8
	}

	return out
}
func PCM16ToBytes(
	pcm []int16,
) []byte {
	out := make(
		[]byte,
		len(pcm)*2,
	)
	for i, sample := range pcm {
		out[i*2] = byte(sample)
		out[i*2+1] = byte(sample >> 8)
	}
	return out
}
func BytesToPCM16(
	data []byte,
) []int16 {

	if len(data)%2 != 0 {
		data = data[:len(data)-1]
	}

	out := make(
		[]int16,
		len(data)/2,
	)

	for i := range out {
		out[i] =
			int16(data[i*2]) |
				int16(data[i*2+1])<<8
	}

	return out
}
func ParseSDP(data []byte) (RemoteMedia, error) {
	var desc sdp.SessionDescription
	if err := desc.Unmarshal(data); err != nil {
		return RemoteMedia{}, err
	}
	var ip net.IP
	if desc.ConnectionInformation != nil && desc.ConnectionInformation.Address != nil {
		ip = net.ParseIP(desc.ConnectionInformation.Address.Address)
	}
	for _, media := range desc.MediaDescriptions {
		if media.MediaName.Media != "audio" {
			continue
		}
		port := int(media.MediaName.Port.Value)
		codecs := parseCodecs(media)
		if len(codecs) == 0 {
			return RemoteMedia{}, errNoAudioCodec
		}
		return RemoteMedia{
			IP:     ip,
			Port:   port,
			Codecs: codecs,
		}, nil
	}
	return RemoteMedia{}, errNoAudioMedia
}
func parseCodecs(
	media *sdp.MediaDescription,
) []neurocall.Codec {
	var result []neurocall.Codec

	for _, format := range media.MediaName.Formats {
		switch format {
		case "0":
			result = append(result, neurocall.Codec{
				Name:        "PCMU",
				PayloadType: 0,
				ClockRate:   8000,
				Channels:    1,
			})

		case "8":
			result = append(result, neurocall.Codec{
				Name:        "PCMA",
				PayloadType: 8,
				ClockRate:   8000,
				Channels:    1,
			})
		}
	}

	return result
}
func ParseSIPMessage(data []byte) (*SIPMessage, error) {
	parts := bytes.SplitN(
		data,
		[]byte("\r\n\r\n"),
		2,
	)

	if len(parts) == 0 {
		return nil, errInvalidSIPMessage
	}

	lines := strings.Split(
		string(parts[0]),
		"\r\n",
	)
	if len(lines) == 0 {
		return nil, errInvalidSIPMessage
	}
	if strings.TrimSpace(lines[0]) == "" {
		return nil, errInvalidSIPMessage
	}
	msg := &SIPMessage{
		StartLine: lines[0],
		Headers:   make(map[string]string),
	}

	for _, line := range lines[1:] {
		idx := strings.IndexByte(line, ':')
		if idx <= 0 {
			continue
		}

		key := strings.ToLower(
			strings.TrimSpace(line[:idx]),
		)

		value := strings.TrimSpace(
			line[idx+1:],
		)

		msg.Headers[key] = value
	}

	if len(parts) == 2 {
		msg.Body = parts[1]
	}

	return msg, nil
}
func BuildSIPResponse(
	req *SIPMessage,
	code int,
	reason string,
	body []byte,
) []byte {
	var b strings.Builder
	fmt.Fprintf(
		&b,
		"SIP/2.0 %d %s\r\n",
		code,
		reason,
	)
	copySIPHeader(&b, req, "Via")
	copySIPHeader(&b, req, "From")
	copySIPHeader(&b, req, "To")
	copySIPHeader(&b, req, "Call-ID")
	copySIPHeader(&b, req, "CSeq")
	if len(body) > 0 {
		b.WriteString(
			"Content-Type: application/sdp\r\n",
		)
	}
	b.WriteString("Content-Length: ")
	b.WriteString(strconv.Itoa(len(body)))
	b.WriteString("\r\n")
	b.WriteString("\r\n")
	b.Write(body)
	return []byte(b.String())
}
func copySIPHeader(
	b *strings.Builder,
	msg *SIPMessage,
	name string,
) {
	value := msg.Headers[strings.ToLower(name)]
	if value == "" {
		return
	}
	fmt.Fprintf(
		b,
		"%s: %s\r\n",
		name,
		value,
	)
}
func DefaultConfig() Config {
	return Config{
		SIP: SIPConfig{
			ListenIP:   "0.0.0.0",
			SIPPort:    DefaultSIPPort,
			RTPMinPort: DefaultRTPMinPort,
			RTPMaxPort: DefaultRTPMaxPort,
		},

		Audio: AudioConfig{
			SampleRate: DefaultAudioSampleRate,
			Channels:   DefaultAudioChannels,
			FrameSize:  DefaultAudioFrameSize,
			BufferSize: 32,
		},

		RTPMinPort: DefaultRTPMinPort,
		RTPMaxPort: DefaultRTPMaxPort,
	}
}
func GetService[T any](
	c *Container,
	name string,
) (T, bool) {
	var zero T

	value, ok := c.Get(name)

	if !ok {
		return zero, false
	}

	result, ok := value.(T)

	return result, ok
}
func BuildSDPAnswer(
	call *SIPCall,
	config SIPConfig,
) []byte {
	// ip := s.MediaIP()
	ip := config.ExternalIP
	if ip == "" {
		ip = config.ListenIP
	}
	var b strings.Builder
	fmt.Fprintf(
		&b,
		"v=0\r\n",
	)
	fmt.Fprintf(
		&b,
		"o=neurocall 1 1 IN IP4 %s\r\n",
		ip,
	)
	fmt.Fprintf(
		&b,
		"s=NeuroCall\r\n",
	)
	fmt.Fprintf(
		&b,
		"c=IN IP4 %s\r\n",
		ip,
	)
	fmt.Fprintf(
		&b,
		"t=0 0\r\n",
	)
	fmt.Fprintf(
		&b,
		"m=audio %d RTP/AVP %d\r\n",
		call.LocalRTPPort,
		call.Codec.PayloadType,
	)
	fmt.Fprintf(
		&b,
		"a=rtpmap:%d %s/%d/%d\r\n",
		call.Codec.PayloadType,
		call.Codec.Name,
		call.Codec.ClockRate,
		call.Codec.Channels,
	)
	return []byte(b.String())
}
func linearToMuLaw(sample int16) byte {
	const (
		bias = 132
	)

	sign := byte(0)

	if sample < 0 {
		sign = 0x80

		if sample == -32768 {
			sample = 32767
		} else {
			sample = -sample
		}
	}

	value := int(sample) + bias

	exponent := 7

	for exponent > 0 {
		if value >= (1 << (exponent + 3)) {
			break
		}

		exponent--
	}

	mantissa :=
		(value >> (exponent + 3)) & 0x0f

	return ^(sign | byte(exponent<<4) | byte(mantissa))
}
func muLawToLinear(value byte) int16 {
	value = ^value

	sign := value & 0x80
	exponent := (value >> 4) & 0x07
	mantissa := value & 0x0f

	sample :=
		((int(mantissa) << 3) + 132) <<
			exponent

	sample -= 132

	if sign != 0 {
		return int16(-sample)
	}

	return int16(sample)
}
func linearToALaw(sample int16) byte {
	sign := byte(0)

	if sample < 0 {
		sign = 0x80

		if sample == -32768 {
			sample = 32767
		} else {
			sample = -sample
		}
	}

	value := int(sample)

	if value > 32635 {
		value = 32635
	}

	exponent := 7

	for exponent > 0 {
		if value >= (1 << (exponent + 3)) {
			break
		}

		exponent--
	}

	var mantissa int

	if exponent == 0 {
		mantissa = value >> 4
	} else {
		mantissa =
			(value >> (exponent + 3)) & 0x0f
	}

	result :=
		sign |
			byte(exponent<<4) |
			byte(mantissa)

	return result ^ 0x55
}
func aLawToLinear(value byte) int16 {
	value ^= 0x55

	sign := value & 0x80
	exponent := (value >> 4) & 0x07
	mantissa := value & 0x0f

	var sample int

	if exponent == 0 {
		sample =
			(int(mantissa) << 4) +
				8
	} else {
		sample =
			((int(mantissa) << 3) + 132) <<
				(exponent - 1)
	}

	if sign != 0 {
		return int16(-sample)
	}

	return int16(sample)
}
func EncoderFor(
	codec neurocall.Codec,
) (neurocall.Encoder, error) {

	switch codec.Name {

	case "PCMU":
		return PCMU{}, nil

	case "PCMA":
		return PCMA{}, nil

	default:
		return nil, fmt.Errorf(
			"unsupported encoder codec: %s",
			codec.Name,
		)
	}
}
func DecoderFor(
	codec neurocall.Codec,
) (neurocall.Decoder, error) {

	switch codec.Name {

	case "PCMU":
		return PCMU{}, nil

	case "PCMA":
		return PCMA{}, nil

	default:
		return nil, fmt.Errorf(
			"unsupported decoder codec: %s",
			codec.Name,
		)
	}
}
func randomUint32() uint32 {
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		return uint32(time.Now().UnixNano())
	}
	return binary.BigEndian.Uint32(b[:])
}
func seqDistance(a, b uint16) int {
	return int(int16(a - b))
}
func concealPCM(
	codec neurocall.Codec,
	frameSamples int,
) []int16 {

	if frameSamples <= 0 {
		frameSamples = 160
	}

	return make([]int16, frameSamples)
}
func seqLess(a, b uint16) bool {
	return int16(a-b) < 0
}
