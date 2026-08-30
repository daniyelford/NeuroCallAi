package core

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func TestPCMUEncodeDecode(t *testing.T) {
	codec := PCMU{}

	input := []int16{
		0,
		1000,
		-1000,
		5000,
		-5000,
		16000,
		-16000,
	}

	encoded := codec.Encode(input)

	if len(encoded) != len(input) {
		t.Fatalf(
			"encoded length mismatch: got %d want %d",
			len(encoded),
			len(input),
		)
	}

	decoded := codec.Decode(encoded)

	if len(decoded) != len(input) {
		t.Fatalf(
			"decoded length mismatch: got %d want %d",
			len(decoded),
			len(input),
		)
	}
}
func TestPCMAEncodeDecode(t *testing.T) {
	codec := PCMA{}

	input := []int16{
		0,
		1000,
		-1000,
		5000,
		-5000,
		16000,
		-16000,
	}

	encoded := codec.Encode(input)

	if len(encoded) != len(input) {
		t.Fatalf(
			"encoded length mismatch: got %d want %d",
			len(encoded),
			len(input),
		)
	}

	decoded := codec.Decode(encoded)

	if len(decoded) != len(input) {
		t.Fatalf(
			"decoded length mismatch: got %d want %d",
			len(decoded),
			len(input),
		)
	}
}
func TestAudioPipeline(t *testing.T) {
	pipeline := NewAudioPipeline(AudioConfig{
		SampleRate: 8000,
		Channels:   1,
		FrameSize:  160,
		BufferSize: 4,
	})

	pipeline.SetEncoder(PCMU{})
	pipeline.SetDecoder(PCMU{})

	pipeline.Start()

	defer pipeline.Close()

	input := make([]int16, 160)

	for i := range input {
		input[i] = int16(i * 100)
	}

	if err := pipeline.Push(input); err != nil {
		t.Fatal(err)
	}

	done := make(chan []int16, 1)

	go func() {
		audio, err := pipeline.Pull()

		if err != nil {
			return
		}

		done <- audio
	}()

	select {

	case output := <-done:

		if len(output) != len(input) {
			t.Fatalf(
				"output length mismatch: got %d want %d",
				len(output),
				len(input),
			)
		}

	case <-time.After(time.Second):
		t.Fatal("pipeline timeout")
	}
}
func TestParseSIPMessage(t *testing.T) {
	raw := []byte(
		"INVITE sip:1000@example.com SIP/2.0\r\n" +
			"Via: SIP/2.0/UDP 127.0.0.1:5060\r\n" +
			"From: <sip:1001@example.com>\r\n" +
			"To: <sip:1000@example.com>\r\n" +
			"Call-ID: test-call-123\r\n" +
			"CSeq: 1 INVITE\r\n" +
			"Content-Length: 4\r\n" +
			"\r\n" +
			"test",
	)

	msg, err := ParseSIPMessage(raw)
	if err != nil {
		t.Fatal(err)
	}

	if msg.StartLine != "INVITE sip:1000@example.com SIP/2.0" {
		t.Fatalf("unexpected start line: %q", msg.StartLine)
	}

	if msg.Method() != "INVITE" {
		t.Fatalf("unexpected method: %q", msg.Method())
	}

	if msg.Headers["call-id"] != "test-call-123" {
		t.Fatalf(
			"unexpected Call-ID: %q",
			msg.Headers["call-id"],
		)
	}

	if string(msg.Body) != "test" {
		t.Fatalf(
			"unexpected body: %q",
			string(msg.Body),
		)
	}
}
func TestParseSIPMessageHeaderCaseInsensitive(t *testing.T) {
	raw := []byte(
		"OPTIONS sip:test SIP/2.0\r\n" +
			"CALL-ID: abc\r\n" +
			"Content-Length: 0\r\n" +
			"\r\n",
	)

	msg, err := ParseSIPMessage(raw)
	if err != nil {
		t.Fatal(err)
	}

	if msg.Headers["call-id"] != "abc" {
		t.Fatalf(
			"expected case-insensitive Call-ID lookup",
		)
	}

	if msg.Method() != "OPTIONS" {
		t.Fatalf("unexpected method: %s", msg.Method())
	}
}
func TestParseSIPMessageInvalid(t *testing.T) {
	tests := [][]byte{
		{},
		[]byte("\r\n\r\n"),
	}
	for _, input := range tests {
		if _, err := ParseSIPMessage(input); err == nil {
			t.Fatal("expected parse error")
		}
	}
}
func TestBuildSIPResponse(t *testing.T) {
	req := &SIPMessage{
		StartLine: "INVITE sip:test SIP/2.0",
		Headers: map[string]string{
			"via":     "SIP/2.0/UDP 127.0.0.1:5060",
			"from":    "<sip:1001@example.com>",
			"to":      "<sip:1000@example.com>",
			"call-id": "call-123",
			"cseq":    "1 INVITE",
		},
	}

	body := []byte("hello")

	response := BuildSIPResponse(
		req,
		200,
		"OK",
		body,
	)

	text := string(response)

	checks := []string{
		"SIP/2.0 200 OK\r\n",
		"Via: SIP/2.0/UDP 127.0.0.1:5060\r\n",
		"From: <sip:1001@example.com>\r\n",
		"To: <sip:1000@example.com>\r\n",
		"Call-ID: call-123\r\n",
		"CSeq: 1 INVITE\r\n",
		"Content-Type: application/sdp\r\n",
		"Content-Length: 5\r\n",
		"\r\n",
		"hello",
	}

	for _, expected := range checks {
		if !strings.Contains(text, expected) {
			t.Fatalf(
				"response missing %q:\n%s",
				expected,
				text,
			)
		}
	}
}
func TestCodecRegistryNegotiation(t *testing.T) {
	registry := NewCodecRegistry()

	remote := []neurocall.Codec{
		{
			Name:        "PCMA",
			PayloadType: 8,
			ClockRate:   8000,
			Channels:    1,
		},
		{
			Name:        "PCMU",
			PayloadType: 0,
			ClockRate:   8000,
			Channels:    1,
		},
	}

	codec, err := registry.Negotiate(remote)
	if err != nil {
		t.Fatal(err)
	}

	if codec.Name == "" {
		t.Fatal("expected negotiated codec")
	}
}
func TestCodecRegistryNoCommonCodec(t *testing.T) {
	registry := NewCodecRegistry()

	remote := []neurocall.Codec{
		{
			Name:        "OPUS",
			PayloadType: 111,
			ClockRate:   48000,
			Channels:    2,
		},
	}

	_, err := registry.Negotiate(remote)

	if err != neurocall.ErrNoCommonCodec {
		t.Fatalf(
			"expected ErrNoCommonCodec, got %v",
			err,
		)
	}
}
func TestParseSDPPCMU(t *testing.T) {
	raw := []byte(
		"v=0\r\n" +
			"o=- 1 1 IN IP4 192.168.1.10\r\n" +
			"s=Test\r\n" +
			"c=IN IP4 192.168.1.10\r\n" +
			"t=0 0\r\n" +
			"m=audio 4000 RTP/AVP 0\r\n" +
			"a=rtpmap:0 PCMU/8000/1\r\n",
	)

	media, err := ParseSDP(raw)
	if err != nil {
		t.Fatal(err)
	}

	if media.IP == nil {
		t.Fatal("expected remote IP")
	}

	if !media.IP.Equal(net.ParseIP("192.168.1.10")) {
		t.Fatalf(
			"unexpected IP: %v",
			media.IP,
		)
	}

	if media.Port != 4000 {
		t.Fatalf(
			"unexpected port: %d",
			media.Port,
		)
	}

	if len(media.Codecs) != 1 {
		t.Fatalf(
			"expected 1 codec, got %d",
			len(media.Codecs),
		)
	}

	codec := media.Codecs[0]

	if codec.Name != "PCMU" {
		t.Fatalf(
			"unexpected codec: %s",
			codec.Name,
		)
	}

	if codec.PayloadType != 0 {
		t.Fatalf(
			"unexpected payload type: %d",
			codec.PayloadType,
		)
	}
}
func TestParseSDPPCMA(t *testing.T) {
	raw := []byte(
		"v=0\r\n" +
			"o=- 1 1 IN IP4 10.0.0.5\r\n" +
			"s=Test\r\n" +
			"c=IN IP4 10.0.0.5\r\n" +
			"t=0 0\r\n" +
			"m=audio 5000 RTP/AVP 8\r\n" +
			"a=rtpmap:8 PCMA/8000/1\r\n",
	)

	media, err := ParseSDP(raw)
	if err != nil {
		t.Fatal(err)
	}

	if media.Port != 5000 {
		t.Fatalf(
			"unexpected port: %d",
			media.Port,
		)
	}

	if len(media.Codecs) != 1 {
		t.Fatalf(
			"expected 1 codec, got %d",
			len(media.Codecs),
		)
	}

	if media.Codecs[0].Name != "PCMA" {
		t.Fatalf(
			"unexpected codec: %s",
			media.Codecs[0].Name,
		)
	}
}
func TestParseSDPMultipleCodecs(t *testing.T) {
	raw := []byte(
		"v=0\r\n" +
			"o=- 1 1 IN IP4 127.0.0.1\r\n" +
			"s=Test\r\n" +
			"c=IN IP4 127.0.0.1\r\n" +
			"t=0 0\r\n" +
			"m=audio 6000 RTP/AVP 0 8\r\n",
	)

	media, err := ParseSDP(raw)
	if err != nil {
		t.Fatal(err)
	}

	if len(media.Codecs) != 2 {
		t.Fatalf(
			"expected 2 codecs, got %d",
			len(media.Codecs),
		)
	}

	if media.Codecs[0].Name != "PCMU" {
		t.Fatalf("expected PCMU first")
	}

	if media.Codecs[1].Name != "PCMA" {
		t.Fatalf("expected PCMA second")
	}
}
func TestParseSDPNoAudio(t *testing.T) {
	raw := []byte(
		"v=0\r\n" +
			"o=- 1 1 IN IP4 127.0.0.1\r\n" +
			"s=Test\r\n" +
			"c=IN IP4 127.0.0.1\r\n" +
			"t=0 0\r\n" +
			"m=video 6000 RTP/AVP 96\r\n",
	)

	_, err := ParseSDP(raw)

	if err != errNoAudioMedia {
		t.Fatalf(
			"expected errNoAudioMedia, got %v",
			err,
		)
	}
}
func TestParseSDPNoSupportedCodec(t *testing.T) {
	raw := []byte(
		"v=0\r\n" +
			"o=- 1 1 IN IP4 127.0.0.1\r\n" +
			"s=Test\r\n" +
			"c=IN IP4 127.0.0.1\r\n" +
			"t=0 0\r\n" +
			"m=audio 6000 RTP/AVP 111\r\n" +
			"a=rtpmap:111 opus/48000/2\r\n",
	)

	_, err := ParseSDP(raw)

	if err != errNoAudioCodec {
		t.Fatalf(
			"expected errNoAudioCodec, got %v",
			err,
		)
	}
}
func TestParseSDPInvalid(t *testing.T) {
	_, err := ParseSDP(
		[]byte("this is not SDP"),
	)

	if err == nil {
		t.Fatal("expected SDP parse error")
	}
}
func TestRTPPacketMarshalUnmarshal(t *testing.T) {
	original := &RTPPacket{
		Header: RTPHeader{
			Version:        2,
			PayloadType:    0,
			SequenceNumber: 1234,
			Timestamp:      160,
			SSRC:           0x12345678,
			Marker:         true,
		},
		Payload: []byte{1, 2, 3, 4, 5},
	}

	data, err := original.Marshal()
	if err != nil {
		t.Fatal(err)
	}

	if len(data) != 12+len(original.Payload) {
		t.Fatalf(
			"unexpected packet length: got %d want %d",
			len(data),
			12+len(original.Payload),
		)
	}

	decoded := &RTPPacket{}

	if err := decoded.Unmarshal(data); err != nil {
		t.Fatal(err)
	}

	if decoded.Header.Version != original.Header.Version {
		t.Fatalf("version mismatch")
	}

	if decoded.Header.PayloadType != original.Header.PayloadType {
		t.Fatalf("payload type mismatch")
	}

	if decoded.Header.SequenceNumber != original.Header.SequenceNumber {
		t.Fatalf("sequence mismatch")
	}

	if decoded.Header.Timestamp != original.Header.Timestamp {
		t.Fatalf("timestamp mismatch")
	}

	if decoded.Header.SSRC != original.Header.SSRC {
		t.Fatalf("SSRC mismatch")
	}

	if decoded.Header.Marker != original.Header.Marker {
		t.Fatalf("marker mismatch")
	}

	if !bytes.Equal(decoded.Payload, original.Payload) {
		t.Fatalf("payload mismatch")
	}
}
func TestRTPPacketInvalidTooShort(t *testing.T) {
	packet := &RTPPacket{}

	err := packet.Unmarshal(
		[]byte{0x80, 0x00},
	)

	if err == nil {
		t.Fatal("expected invalid RTP packet error")
	}
}
func TestRTPPacketInvalidVersion(t *testing.T) {
	packet := &RTPPacket{}

	data := make([]byte, 12)
	data[0] = 0x40

	err := packet.Unmarshal(data)

	if err == nil {
		t.Fatal("expected invalid RTP version error")
	}
}
func TestRTPPacketMarshalInvalidVersion(t *testing.T) {
	packet := &RTPPacket{
		Header: RTPHeader{
			Version: 1,
		},
	}

	_, err := packet.Marshal()

	if err == nil {
		t.Fatal("expected invalid RTP version error")
	}
}
func TestRTPPacketCSRCUnsupported(t *testing.T) {
	packet := &RTPPacket{
		Header: RTPHeader{
			Version:   2,
			CSRCCount: 1,
		},
	}

	_, err := packet.Marshal()

	if err == nil {
		t.Fatal("expected CSRC error")
	}
}
func TestRTPPacketEmptyPayload(t *testing.T) {
	packet := &RTPPacket{
		Header: RTPHeader{
			Version:        2,
			PayloadType:    0,
			SequenceNumber: 1,
			Timestamp:      160,
			SSRC:           123,
		},
	}

	data, err := packet.Marshal()
	if err != nil {
		t.Fatal(err)
	}

	decoded := &RTPPacket{}

	if err := decoded.Unmarshal(data); err != nil {
		t.Fatal(err)
	}

	if len(decoded.Payload) != 0 {
		t.Fatalf(
			"expected empty payload, got %d bytes",
			len(decoded.Payload),
		)
	}
}
func TestRTPPacketExtension(t *testing.T) {
	packet := &RTPPacket{}
	data := make([]byte, 20)
	data[0] = 0x90
	data[14] = 0
	data[15] = 2
	err := packet.Unmarshal(data)
	if err == nil {
		t.Fatal("expected invalid extension error")
	}
}
func TestRTPSessionWriteRead(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	receiver, err := net.ListenUDP(
		"udp",
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: 0,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer receiver.Close()

	receiverAddr := receiver.LocalAddr().(*net.UDPAddr)

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		receiverAddr.IP,
		receiverAddr.Port,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	payload := []byte{
		1, 2, 3, 4, 5,
	}

	if err := session.Write(payload, 1); err != nil {
		t.Fatal(err)
	}

	buf := make([]byte, 2048)

	n, _, err := receiver.ReadFromUDP(buf)
	if err != nil {
		t.Fatal(err)
	}

	packet := &RTPPacket{}

	if err := packet.Unmarshal(buf[:n]); err != nil {
		t.Fatal(err)
	}

	if packet.Header.Version != 2 {
		t.Fatalf(
			"unexpected RTP version: %d",
			packet.Header.Version,
		)
	}

	if packet.Header.PayloadType != codec.PayloadType {
		t.Fatalf(
			"unexpected payload type: %d",
			packet.Header.PayloadType,
		)
	}

	if len(packet.Payload) != len(payload) {
		t.Fatalf(
			"unexpected payload length: got %d want %d",
			len(packet.Payload),
			len(payload),
		)
	}
}
func TestRTPSessionRead(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	sender, err := net.ListenUDP(
		"udp",
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: 0,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer sender.Close()

	senderAddr := sender.LocalAddr().(*net.UDPAddr)

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		senderAddr.IP,
		senderAddr.Port,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	sessionAddr := session.conn.LocalAddr().(*net.UDPAddr)

	packet := &RTPPacket{
		Header: RTPHeader{
			Version:        2,
			PayloadType:    0,
			SequenceNumber: 42,
			Timestamp:      160,
			SSRC:           12345,
		},
		Payload: []byte{
			10, 20, 30,
		},
	}

	data, err := packet.Marshal()
	if err != nil {
		t.Fatal(err)
	}

	if _, err := sender.WriteToUDP(
		data,
		sessionAddr,
	); err != nil {
		t.Fatal(err)
	}

	received, err := session.Read()
	if err != nil {
		t.Fatal(err)
	}

	if received.Header.SequenceNumber != 42 {
		t.Fatalf(
			"unexpected sequence: %d",
			received.Header.SequenceNumber,
		)
	}

	if received.Header.Timestamp != 160 {
		t.Fatalf(
			"unexpected timestamp: %d",
			received.Header.Timestamp,
		)
	}

	if received.Header.SSRC != 12345 {
		t.Fatalf(
			"unexpected SSRC: %d",
			received.Header.SSRC,
		)
	}

	if len(received.Payload) != 3 {
		t.Fatalf(
			"unexpected payload length: %d",
			len(received.Payload),
		)
	}
}
func TestRTPSessionClose(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9999,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	err = session.Write([]byte{1}, 1)

	if err == nil {
		t.Fatal("expected error after session close")
	}
}
func TestSTTWorker(t *testing.T) {
	bus := NewEventBus()
	engine := NewSTTEngine(NewFakeSTT())
	worker, err := NewSTTWorker(
		engine,
		bus,
		"test-call",
		4,
	)
	if err != nil {
		t.Fatal(err)
	}
	received := make(chan TranscriptEvent, 1)
	bus.Subscribe(
		EventTranscript,
		func(event Event) {

			data, ok :=
				event.Data.(TranscriptEvent)

			if !ok {
				return
			}

			received <- data
		},
	)

	worker.Start()

	segment := neurocall.AudioSegment{
		Data:       []int16{100, 200, 300, 400},
		SampleRate: 8000,
		Channels:   1,
	}

	if err := worker.Push(segment); err != nil {
		t.Fatal(err)
	}

	select {

	case event := <-received:

		if event.CallID != "test-call" {
			t.Fatalf(
				"unexpected call ID: %s",
				event.CallID,
			)
		}

		if event.Transcript.Text != "hello from fake stt" {
			t.Fatalf(
				"unexpected transcript: %s",
				event.Transcript.Text,
			)
		}

	case <-time.After(time.Second):
		t.Fatal("timeout waiting for transcript")
	}

	worker.Stop()
	engine.Close()
}
func TestRTPSequenceTrackerWrap(t *testing.T) {
	tracker := &RTPSequenceTracker{}

	tracker.Update(65534)
	tracker.Update(65535)
	tracker.Update(0)
	tracker.Update(1)

	stats := tracker.Stats()

	if stats.LostPackets != 0 {
		t.Fatalf(
			"expected 0 lost packets, got %d",
			stats.LostPackets,
		)
	}
}
func TestRTPSequenceTrackerLoss1(t *testing.T) {
	tracker := &RTPSequenceTracker{}

	tracker.Update(100)
	tracker.Update(101)
	tracker.Update(103)

	stats := tracker.Stats()

	if stats.LostPackets != 1 {
		t.Fatalf(
			"expected 1 lost packet, got %d",
			stats.LostPackets,
		)
	}
}
func TestRTPSequenceTrackerDuplicate(t *testing.T) {
	tracker := &RTPSequenceTracker{}

	tracker.Update(100)
	tracker.Update(100)

	stats := tracker.Stats()

	if stats.Duplicates != 1 {
		t.Fatalf(
			"expected 1 duplicate, got %d",
			stats.Duplicates,
		)
	}
}
func TestRTPSequenceTrackerOutOfOrder(t *testing.T) {
	tracker := &RTPSequenceTracker{}

	tracker.Update(100)
	tracker.Update(102)
	tracker.Update(101)

	stats := tracker.Stats()

	if stats.OutOfOrder != 1 {
		t.Fatalf(
			"expected 1 out-of-order packet, got %d",
			stats.OutOfOrder,
		)
	}
}
func TestRTPJitterBufferBasic(t *testing.T) {
	buffer := NewRTPJitterBuffer(
		10,
		100*time.Millisecond,
	)

	packet := &RTPPacket{}
	packet.Header.SequenceNumber = 100

	if err := buffer.Push(packet); err != nil {
		t.Fatalf("Push failed: %v", err)
	}

	got, ok := buffer.Pop()

	if !ok {
		t.Fatal("expected packet to be available")
	}

	if got == nil {
		t.Fatal("expected non-nil packet")
	}

	if got.Header.SequenceNumber != 100 {
		t.Fatalf(
			"expected sequence 100, got %d",
			got.Header.SequenceNumber,
		)
	}
}
func TestRTPJitterBufferDuplicate(t *testing.T) {
	buffer := NewRTPJitterBuffer(
		10,
		100*time.Millisecond,
	)

	packet := &RTPPacket{}
	packet.Header.SequenceNumber = 100

	if err := buffer.Push(packet); err != nil {
		t.Fatalf("first Push failed: %v", err)
	}

	if err := buffer.Push(packet); err != nil {
		t.Fatalf("duplicate Push failed: %v", err)
	}

	got, ok := buffer.Pop()

	if !ok || got == nil {
		t.Fatal("expected packet")
	}

	if got.Header.SequenceNumber != 100 {
		t.Fatalf(
			"expected 100, got %d",
			got.Header.SequenceNumber,
		)
	}

	got, ok = buffer.Pop()

	if ok || got != nil {
		t.Fatal("expected buffer to be empty")
	}
}
func TestRTPJitterBufferSequenceWrap(t *testing.T) {
	buffer := NewRTPJitterBuffer(
		10,
		100*time.Millisecond,
	)

	p65535 := &RTPPacket{}
	p65535.Header.SequenceNumber = 65535

	p0 := &RTPPacket{}
	p0.Header.SequenceNumber = 0

	if err := buffer.Push(p65535); err != nil {
		t.Fatalf("Push 65535 failed: %v", err)
	}

	if err := buffer.Push(p0); err != nil {
		t.Fatalf("Push 0 failed: %v", err)
	}

	got, ok := buffer.Pop()

	if !ok || got == nil {
		t.Fatal("expected packet 65535")
	}

	if got.Header.SequenceNumber != 65535 {
		t.Fatalf(
			"expected 65535, got %d",
			got.Header.SequenceNumber,
		)
	}

	got, ok = buffer.Pop()

	if !ok || got == nil {
		t.Fatal("expected packet 0")
	}

	if got.Header.SequenceNumber != 0 {
		t.Fatalf(
			"expected 0, got %d",
			got.Header.SequenceNumber,
		)
	}
}
func TestNewCallSession(t *testing.T) {
	call := &SIPCall{
		CallID: "test-call-1",
	}

	session, err := NewCallSession(call, NewEventBus())

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if session == nil {
		t.Fatal("session is nil")
	}

	if session.ID != "test-call-1" {
		t.Fatalf(
			"expected ID test-call-1, got %s",
			session.ID,
		)
	}

	if session.Call != call {
		t.Fatal("session does not contain original call")
	}

	if session.Events == nil {
		t.Fatal("event bus is nil")
	}

	if session.Memory == nil {
		t.Fatal("memory is nil")
	}

	if session.Tools == nil {
		t.Fatal("tools are nil")
	}
}
func TestCallSessionStart(t *testing.T) {
	call := &SIPCall{
		CallID: "test-call-start",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if session.Running() {
		t.Fatal("session should not be running initially")
	}

	if err := session.Start(); err != nil {
		t.Fatalf("start failed: %v", err)
	}

	if !session.Running() {
		t.Fatal("session should be running")
	}

	if session.Closed() {
		t.Fatal("session should not be closed")
	}
}
func TestCallSessionStartTwice(t *testing.T) {
	call := &SIPCall{
		CallID: "test-call-twice",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatalf(
			"second Start should be harmless: %v",
			err,
		)
	}

	if !session.Running() {
		t.Fatal("session should still be running")
	}
}
func TestCallSessionContext(t *testing.T) {
	call := &SIPCall{
		CallID: "test-context",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	ctx := session.Context()

	if ctx == nil {
		t.Fatal("context is nil")
	}

	select {
	case <-ctx.Done():
		t.Fatal("context should not be cancelled")
	default:
	}
}
func TestCallSessionClose(t *testing.T) {
	call := &SIPCall{
		CallID: "test-close",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	ctx := session.Context()

	if err := session.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	if !session.Closed() {
		t.Fatal("session should be closed")
	}

	if session.Running() {
		t.Fatal("session should not be running")
	}

	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("session context was not cancelled")
	}
}
func TestCallSessionCloseTwice(t *testing.T) {
	call := &SIPCall{
		CallID: "test-close-twice",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatalf(
			"second Close should be harmless: %v",
			err,
		)
	}
}
func TestCallSessionCannotStartAfterClose(t *testing.T) {
	call := &SIPCall{
		CallID: "test-start-after-close",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	err = session.Start()

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}

	if session.Running() {
		t.Fatal("closed session cannot be running")
	}
}
func TestCallSessionPushAudioBeforeStart(t *testing.T) {
	call := &SIPCall{
		CallID: "test-push-before-start",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	err = session.PushAudio([]int16{1, 2, 3})

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestCallSessionPushAudioWithoutPipeline(t *testing.T) {
	call := &SIPCall{
		CallID: "test-no-pipeline",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	err = session.PushAudio([]int16{1, 2, 3})

	if err == nil {
		t.Fatal("expected audio pipeline configuration error")
	}
}
func TestCallSessionPushSTTSegmentWithoutWorker(t *testing.T) {
	call := &SIPCall{
		CallID: "test-no-stt-worker",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	segment := neurocall.AudioSegment{
		Data:       []int16{100, 200, 300},
		SampleRate: 8000,
		Channels:   1,
		Start:      0,
		End:        20 * time.Millisecond,
		Final:      true,
	}

	err = session.PushSTTSegment(segment)

	if err == nil {
		t.Fatal("expected STT worker configuration error")
	}
}
func TestCallSessionSetComponents(t *testing.T) {
	call := &SIPCall{
		CallID: "test-components",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	stt := NewSTTEngine(nil)
	tts := NewTTSEngine(nil)

	session.SetSTT(stt)
	session.SetTTS(tts)

	if session.STT != stt {
		t.Fatal("STT was not set correctly")
	}

	if session.TTS != tts {
		t.Fatal("TTS was not set correctly")
	}
}
func TestCallSessionSetMemoryAndTools(t *testing.T) {
	call := &SIPCall{
		CallID: "test-memory-tools",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	memory := NewMemory()
	tools := NewToolRegistry()

	session.SetMemory(memory)
	session.SetTools(tools)

	if session.Memory != memory {
		t.Fatal("memory was not set correctly")
	}

	if session.Tools != tools {
		t.Fatal("tools were not set correctly")
	}
}
func TestCallSessionSetSTTWorker(t *testing.T) {
	call := &SIPCall{
		CallID: "test-stt-worker",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	engine := NewSTTEngine(NewFakeSTT())

	worker, err := NewSTTWorker(
		engine,
		session.Events,
		session.ID,
		4,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.SetSTTWorker(worker); err != nil {
		t.Fatal(err)
	}

	if session.GetSTTWorker() != worker {
		t.Fatal("STT worker was not set correctly")
	}
}
func TestCallSessionSTTWorkerFlow(t *testing.T) {
	call := &SIPCall{
		CallID: "test-stt-flow",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}
	engine := NewSTTEngine(NewFakeSTT())

	worker, err := NewSTTWorker(
		engine,
		session.Events,
		session.ID,
		4,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.SetSTTWorker(worker); err != nil {
		t.Fatal(err)
	}

	received := make(chan TranscriptEvent, 1)

	session.Events.Subscribe(
		EventTranscript,
		func(event Event) {
			data, ok := event.Data.(TranscriptEvent)
			if !ok {
				return
			}

			select {
			case received <- data:
			default:
			}
		},
	)

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	segment := neurocall.AudioSegment{
		Data:       []int16{100, 200, 300, 400},
		SampleRate: 8000,
		Channels:   1,
		Start:      0,
		End:        20 * time.Millisecond,
		Final:      true,
	}

	if err := session.PushSTTSegment(segment); err != nil {
		t.Fatal(err)
	}

	select {
	case event := <-received:
		if event.CallID != session.ID {
			t.Fatalf(
				"expected call ID %q, got %q",
				session.ID,
				event.CallID,
			)
		}

		if event.Transcript.Text != "hello from fake stt" {
			t.Fatalf(
				"unexpected transcript: %q",
				event.Transcript.Text,
			)
		}

	case <-time.After(time.Second):
		t.Fatal("timeout waiting for transcript")
	}

	_ = session.Close()
}
func TestCallSessionPushSTTSegmentAfterClose(t *testing.T) {
	call := &SIPCall{
		CallID: "test-stt-after-close",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	engine := NewSTTEngine(NewFakeSTT())

	worker, err := NewSTTWorker(
		engine,
		session.Events,
		session.ID,
		4,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.SetSTTWorker(worker); err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	segment := neurocall.AudioSegment{
		Data:       []int16{1, 2, 3, 4},
		SampleRate: 8000,
		Channels:   1,
		Final:      true,
	}

	err = session.PushSTTSegment(segment)

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestCallSessionWithAudioPipeline(t *testing.T) {
	call := &SIPCall{
		CallID: "test-audio-pipeline",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 4,
	})

	session.Audio = pipeline

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	if !session.Running() {
		t.Fatal("session should be running")
	}

	if err := session.PushAudio(
		[]int16{100, 200, 300, 400},
	); err != nil {
		t.Fatalf(
			"PushAudio failed: %v",
			err,
		)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	if !session.Closed() {
		t.Fatal("session should be closed")
	}
}
func TestCallSessionState(t *testing.T) {
	call := &SIPCall{
		CallID: "test-call",
	}

	session, err := NewCallSession(call, NewEventBus())

	if err != nil {
		t.Fatal(err)
	}

	state := session.State()

	if state.ID != "test-call" {
		t.Fatalf(
			"unexpected ID: %s",
			state.ID,
		)
	}

	if state.Running {
		t.Fatal("session should not be running")
	}

	if state.Closed {
		t.Fatal("session should not be closed")
	}
}
func TestAudioProcessorStart(t *testing.T) {
	bus := NewEventBus()

	vad := &testVAD{
		result: true,
	}

	segmenter := &testSegmenter{}

	worker, err := newTestSTTWorker(bus)
	if err != nil {
		t.Fatal(err)
	}

	processor, err := NewAudioProcessor(
		vad,
		segmenter,
		worker,
		bus,
	)

	if err != nil {
		t.Fatal(err)
	}

	if processor.Running() {
		t.Fatal("processor should not be running")
	}

	if err := processor.Start(); err != nil {
		t.Fatal(err)
	}

	if !processor.Running() {
		t.Fatal("processor should be running")
	}

	if err := processor.Start(); err != nil {
		t.Fatal(err)
	}

	if err := processor.Stop(); err != nil {
		t.Fatal(err)
	}

	if !processor.Closed() {
		t.Fatal("processor should be closed")
	}
}
func TestAudioProcessorSilence(t *testing.T) {

	bus := NewEventBus()

	vad := &testVAD{
		result: false,
	}

	segmenter := &testSegmenter{
		segments: []neurocall.AudioSegment{
			{
				Data:       []int16{1, 2, 3},
				SampleRate: 8000,
				Channels:   1,
			},
		},
	}

	worker, err := newTestSTTWorker(bus)
	if err != nil {
		t.Fatal(err)
	}

	processor, err := NewAudioProcessor(
		vad,
		segmenter,
		worker,
		bus,
	)

	if err != nil {
		t.Fatal(err)
	}

	if err := processor.Start(); err != nil {
		t.Fatal(err)
	}

	defer processor.Stop()

	frame := neurocall.AudioFrame{
		Data:       []int16{1, 2, 3, 4},
		Timestamp:  1,
		SampleRate: 8000,
		Channels:   1,
	}

	if err := processor.ProcessFrame(frame); err != nil {
		t.Fatal(err)
	}

	segmenter.mu.Lock()
	calls := segmenter.processCalls
	segmenter.mu.Unlock()

	if calls != 0 {
		t.Fatalf(
			"segmenter should not be called for silence, got %d",
			calls,
		)
	}
}
func TestAudioProcessorSpeech(t *testing.T) {

	bus := NewEventBus()

	vad := &testVAD{
		result: true,
	}

	segment := neurocall.AudioSegment{
		Data:       []int16{1, 2, 3, 4},
		SampleRate: 8000,
		Channels:   1,
		Final:      true,
	}

	segmenter := &testSegmenter{
		segments: []neurocall.AudioSegment{
			segment,
		},
	}

	worker, err := newTestSTTWorker(bus)
	if err != nil {
		t.Fatal(err)
	}

	processor, err := NewAudioProcessor(
		vad,
		segmenter,
		worker,
		bus,
	)

	if err != nil {
		t.Fatal(err)
	}

	if err := processor.Start(); err != nil {
		t.Fatal(err)
	}

	defer processor.Stop()

	frame := neurocall.AudioFrame{
		Data:       []int16{10, 20, 30, 40},
		Timestamp:  10,
		SampleRate: 8000,
		Channels:   1,
	}

	if err := processor.ProcessFrame(frame); err != nil {
		t.Fatal(err)
	}

	segmenter.mu.Lock()
	calls := segmenter.processCalls
	segmenter.mu.Unlock()

	if calls != 1 {
		t.Fatalf(
			"expected segmenter to be called once, got %d",
			calls,
		)
	}
}
func TestAudioProcessorFlush(t *testing.T) {

	bus := NewEventBus()

	vad := &testVAD{
		result: true,
	}

	segmenter := &testSegmenter{
		segments: []neurocall.AudioSegment{
			{
				Data:       []int16{1, 2, 3},
				SampleRate: 8000,
				Channels:   1,
				Final:      true,
			},
		},
	}

	worker, err := newTestSTTWorker(bus)
	if err != nil {
		t.Fatal(err)
	}

	processor, err := NewAudioProcessor(
		vad,
		segmenter,
		worker,
		bus,
	)

	if err != nil {
		t.Fatal(err)
	}

	if err := processor.Start(); err != nil {
		t.Fatal(err)
	}

	defer processor.Stop()

	if err := processor.Flush(); err != nil {
		t.Fatal(err)
	}

	segmenter.mu.Lock()
	calls := segmenter.flushCalls
	segmenter.mu.Unlock()

	if calls != 1 {
		t.Fatalf(
			"expected flush once, got %d",
			calls,
		)
	}
}
func TestAudioProcessorReset(t *testing.T) {

	bus := NewEventBus()

	vad := &testVAD{
		result: true,
	}

	segmenter := &testSegmenter{}

	worker, err := newTestSTTWorker(bus)
	if err != nil {
		t.Fatal(err)
	}

	processor, err := NewAudioProcessor(
		vad,
		segmenter,
		worker,
		bus,
	)

	if err != nil {
		t.Fatal(err)
	}

	if err := processor.Start(); err != nil {
		t.Fatal(err)
	}

	defer processor.Stop()

	if err := processor.ResetSegmenter(); err != nil {
		t.Fatal(err)
	}

	segmenter.mu.Lock()
	calls := segmenter.resetCalls
	segmenter.mu.Unlock()

	if calls != 1 {
		t.Fatalf(
			"expected reset once, got %d",
			calls,
		)
	}
}
func TestAudioProcessorInvalidFrame(t *testing.T) {

	bus := NewEventBus()

	vad := &testVAD{
		result: true,
	}

	segmenter := &testSegmenter{}

	worker, err := newTestSTTWorker(bus)
	if err != nil {
		t.Fatal(err)
	}

	processor, err := NewAudioProcessor(
		vad,
		segmenter,
		worker,
		bus,
	)

	if err != nil {
		t.Fatal(err)
	}

	if err := processor.Start(); err != nil {
		t.Fatal(err)
	}

	defer processor.Stop()

	tests := []struct {
		name  string
		frame neurocall.AudioFrame
	}{
		{
			name: "empty data",
			frame: neurocall.AudioFrame{
				SampleRate: 8000,
				Channels:   1,
			},
		},
		{
			name: "invalid sample rate",
			frame: neurocall.AudioFrame{
				Data:       []int16{1, 2},
				SampleRate: 0,
				Channels:   1,
			},
		},
		{
			name: "invalid channels",
			frame: neurocall.AudioFrame{
				Data:       []int16{1, 2},
				SampleRate: 8000,
				Channels:   0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := processor.ProcessFrame(tt.frame); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}
func TestAudioProcessorCannotProcessAfterStop(t *testing.T) {

	bus := NewEventBus()

	vad := &testVAD{
		result: true,
	}

	segmenter := &testSegmenter{}

	worker, err := newTestSTTWorker(bus)
	if err != nil {
		t.Fatal(err)
	}

	processor, err := NewAudioProcessor(
		vad,
		segmenter,
		worker,
		bus,
	)

	if err != nil {
		t.Fatal(err)
	}

	if err := processor.Start(); err != nil {
		t.Fatal(err)
	}

	if err := processor.Stop(); err != nil {
		t.Fatal(err)
	}

	frame := neurocall.AudioFrame{
		Data:       []int16{1, 2, 3},
		SampleRate: 8000,
		Channels:   1,
	}

	if err := processor.ProcessFrame(frame); err == nil {
		t.Fatal("expected error after stop")
	}
}
func TestAudioProcessorContext(t *testing.T) {

	bus := NewEventBus()

	vad := &testVAD{
		result: true,
	}

	segmenter := &testSegmenter{}

	worker, err := newTestSTTWorker(bus)
	if err != nil {
		t.Fatal(err)
	}

	processor, err := NewAudioProcessor(
		vad,
		segmenter,
		worker,
		bus,
	)

	if err != nil {
		t.Fatal(err)
	}

	if err := processor.Start(); err != nil {
		t.Fatal(err)
	}

	ctx := processor.Context()

	select {
	case <-ctx.Done():
		t.Fatal("context should still be alive")
	default:
	}

	if err := processor.Stop(); err != nil {
		t.Fatal(err)
	}

	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("context was not cancelled")
	}
}
func TestCallManagerRemove(t *testing.T) {
	manager := NewCallManager(10000, 10100)

	call := &SIPCall{
		CallID: "remove-test",
	}

	if err := manager.Add(call); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if _, err := manager.Get(call.CallID); err != nil {
		t.Fatalf("call was not registered: %v", err)
	}

	if err := manager.Remove(call.CallID); err != nil {
		t.Fatalf("Remove failed: %v", err)
	}

	if _, err := manager.Get(call.CallID); err == nil {
		t.Fatal("call still exists after Remove")
	}
}
func TestCallManagerRemoveWithoutSession(t *testing.T) {
	manager := NewCallManager(10000, 10100)
	call := &SIPCall{
		CallID: "no-session-test",
	}

	if err := manager.Add(call); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if err := manager.Remove(call.CallID); err != nil {
		t.Fatalf("Remove failed: %v", err)
	}

	if _, err := manager.Get(call.CallID); err == nil {
		t.Fatal("call still exists after Remove")
	}
}
func TestCallManagerRemoveTwice(t *testing.T) {
	manager := NewCallManager(10000, 10100)
	call := &SIPCall{
		CallID: "double-remove-test",
	}

	if err := manager.Add(call); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if err := manager.Remove(call.CallID); err != nil {
		t.Fatalf("first Remove failed: %v", err)
	}

	err := manager.Remove(call.CallID)

	if !errors.Is(err, neurocall.ErrCallNotFound) {
		t.Fatalf(
			"expected ErrCallNotFound on second Remove, got %v",
			err,
		)
	}
}
func TestCallManagerRemoveFullLifecycle(t *testing.T) {
	manager := NewCallManager(10000, 10100)
	call := &SIPCall{
		CallID: "full-lifecycle-test",
	}

	if err := manager.Add(call); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	session, err := manager.CreateSession(call)
	if err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}

	if session.Closed() {
		t.Fatal("session unexpectedly closed")
	}

	if _, err := manager.Get(call.CallID); err != nil {
		t.Fatalf("call missing before Remove: %v", err)
	}

	if _, err := manager.GetSession(call.CallID); err != nil {
		t.Fatalf("session missing before Remove: %v", err)
	}

	if err := manager.Remove(call.CallID); err != nil {
		t.Fatalf("Remove failed: %v", err)
	}

	if !session.Closed() {
		t.Fatal("session was not closed")
	}

	if _, err := manager.Get(call.CallID); err == nil {
		t.Fatal("call still exists")
	}

	if _, err := manager.GetSession(call.CallID); err == nil {
		t.Fatal("session still exists")
	}
}
func TestCallManagerRemoveReleasesRTPPort(t *testing.T) {
	manager := NewCallManager(10000, 10000)

	call := &SIPCall{
		CallID: "port-release-test",
	}

	if err := manager.Add(call); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	port, err := manager.ports.Allocate()
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}

	call.mu.Lock()
	call.LocalRTPPort = port
	call.mu.Unlock()

	if port != 10000 {
		t.Fatalf("expected port 10000, got %d", port)
	}

	if err := manager.Remove(call.CallID); err != nil {
		t.Fatalf("Remove failed: %v", err)
	}

	reusedPort, err := manager.ports.Allocate()
	if err != nil {
		t.Fatalf(
			"port was not released after Remove: %v",
			err,
		)
	}

	if reusedPort != port {
		t.Fatalf(
			"expected released port %d to be reused, got %d",
			port,
			reusedPort,
		)
	}
}
func TestCallManagerRemoveWithoutRTPPort(t *testing.T) {
	manager := NewCallManager(10000, 10010)

	call := &SIPCall{
		CallID: "no-port-test",
	}

	if err := manager.Add(call); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if err := manager.Remove(call.CallID); err != nil {
		t.Fatalf("Remove failed: %v", err)
	}

	if _, err := manager.Get(call.CallID); err == nil {
		t.Fatal("call still exists after Remove")
	}
}
func TestCallManagerRemoveClosesSession(t *testing.T) {
	manager := NewCallManager(10000, 10010)

	call := &SIPCall{
		CallID: "session-close-test",
	}

	if err := manager.Add(call); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	session, err := manager.CreateSession(call)
	if err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}

	if session.Closed() {
		t.Fatal("session is already closed")
	}

	if err := manager.Remove(call.CallID); err != nil {
		t.Fatalf("Remove failed: %v", err)
	}

	if !session.Closed() {
		t.Fatal("session was not closed")
	}
}
func TestCallManagerConcurrentAddRemove(t *testing.T) {
	manager := NewCallManager(10000, 10100)

	const count = 50

	var wg sync.WaitGroup
	wg.Add(count)

	for i := 0; i < count; i++ {
		go func(i int) {
			defer wg.Done()

			callID := fmt.Sprintf("concurrent-call-%d", i)

			call := &SIPCall{
				CallID: callID,
			}

			if err := manager.Add(call); err != nil {
				t.Errorf("Add(%s) failed: %v", callID, err)
				return
			}

			if _, err := manager.Get(callID); err != nil {
				t.Errorf("Get(%s) failed: %v", callID, err)
				return
			}

			if err := manager.Remove(callID); err != nil {
				t.Errorf("Remove(%s) failed: %v", callID, err)
			}
		}(i)
	}

	wg.Wait()

	if calls := manager.List(); len(calls) != 0 {
		t.Fatalf(
			"expected manager to be empty, got %d calls",
			len(calls),
		)
	}
}
func TestCallManagerConcurrentSessions(t *testing.T) {
	manager := NewCallManager(10000, 10100)

	const count = 50

	var wg sync.WaitGroup
	wg.Add(count)

	for i := 0; i < count; i++ {
		go func(i int) {
			defer wg.Done()

			callID := fmt.Sprintf("session-call-%d", i)

			call := &SIPCall{
				CallID: callID,
			}

			if err := manager.Add(call); err != nil {
				t.Errorf("Add(%s) failed: %v", callID, err)
				return
			}

			session, err := manager.CreateSession(call)
			if err != nil {
				t.Errorf(
					"CreateSession(%s) failed: %v",
					callID,
					err,
				)
				return
			}

			if session == nil {
				t.Errorf("session is nil for %s", callID)
				return
			}

			if _, err := manager.GetSession(callID); err != nil {
				t.Errorf(
					"GetSession(%s) failed: %v",
					callID,
					err,
				)
				return
			}

			if err := manager.Remove(callID); err != nil {
				t.Errorf(
					"Remove(%s) failed: %v",
					callID,
					err,
				)
			}
		}(i)
	}

	wg.Wait()

	if calls := manager.List(); len(calls) != 0 {
		t.Fatalf(
			"expected 0 calls, got %d",
			len(calls),
		)
	}
}
func TestSIPCallCloseResourcesEmpty(t *testing.T) {
	call := &SIPCall{
		CallID: "empty-resources",
	}

	if err := call.CloseResources(); err != nil {
		t.Fatalf("CloseResources failed: %v", err)
	}

	if !call.closed {
		t.Fatal("call should be marked closed")
	}

	if call.RTP != nil {
		t.Fatal("RTP should be nil")
	}

	if call.Pipeline != nil {
		t.Fatal("Pipeline should be nil")
	}

	if call.TTS != nil {
		t.Fatal("TTS should be nil")
	}
}
func TestSIPCallCloseResourcesTwice(t *testing.T) {
	call := &SIPCall{
		CallID: "double-close",
	}

	if err := call.CloseResources(); err != nil {
		t.Fatalf("first CloseResources failed: %v", err)
	}

	if err := call.CloseResources(); err != nil {
		t.Fatalf("second CloseResources failed: %v", err)
	}

	if !call.closed {
		t.Fatal("call should remain closed")
	}
}
func TestSIPCallCloseResourcesClearsReferences1(t *testing.T) {
	call := &SIPCall{
		CallID: "clear-resources",
	}
	if err := call.CloseResources(); err != nil {
		t.Fatalf("CloseResources failed: %v", err)
	}

	if call.RTP != nil ||
		call.Pipeline != nil ||
		call.TTS != nil {
		t.Fatal("resources were not cleared")
	}
}
func TestTestClosableResource(t *testing.T) {
	resource := &testClosableResource{}

	if resource.Closed() {
		t.Fatal("resource should initially be open")
	}

	if err := resource.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	if !resource.Closed() {
		t.Fatal("resource should be closed")
	}

	if err := resource.Close(); err != nil {
		t.Fatalf("second Close failed: %v", err)
	}
}
func TestConversationEngineTranscriptToLLM(t *testing.T) {
	bus := NewEventBus()

	fakeLLM := &testLLM{
		response: neurocall.Message{
			Role:    "assistant",
			Content: "Hello! How can I help you?",
		},
	}

	llm := NewLLMEngine(fakeLLM)

	engine := NewConversationEngine(
		llm,
		bus,
	)

	received := make(chan LLMResponseEvent, 1)

	bus.Subscribe(
		EventLLMResponse,
		func(event Event) {
			data, ok := event.Data.(LLMResponseEvent)
			if !ok {
				return
			}

			received <- data
		},
	)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "call-test-1",
			Transcript: neurocall.Transcript{
				Text: "Hello",
			},
		},
	)

	if err != nil {
		t.Fatalf("HandleTranscript failed: %v", err)
	}

	select {
	case response := <-received:
		if response.CallID != "call-test-1" {
			t.Fatalf(
				"unexpected call ID: %q",
				response.CallID,
			)
		}

		if response.Message.Content != "Hello! How can I help you?" {
			t.Fatalf(
				"unexpected response: %q",
				response.Message.Content,
			)
		}

	case <-time.After(time.Second):
		t.Fatal("timeout waiting for LLM response")
	}
}
func TestConversationEngineGetOrCreate(t *testing.T) {
	llm := &testLLM{
		response: neurocall.Message{
			Role:    "assistant",
			Content: "hello",
		},
	}

	engine := NewConversationEngine(
		NewLLMEngine(llm),
		nil,
	)

	c1, err := engine.GetOrCreate("call-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if c1 == nil {
		t.Fatal("conversation is nil")
	}

	c2, err := engine.GetOrCreate("call-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if c1 != c2 {
		t.Fatal("expected same conversation for same call ID")
	}
}
func TestConversationEngineHandleTranscript(t *testing.T) {
	llm := &testLLM{
		response: neurocall.Message{
			Role:    "assistant",
			Content: "Hello! How can I help you?",
		},
	}

	engine := NewConversationEngine(
		NewLLMEngine(llm),
		nil,
	)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "call-123",
			Transcript: neurocall.Transcript{
				Text: "hello",
			},
		},
	)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if llm.calls != 1 {
		t.Fatalf(
			"expected LLM to be called once, got %d",
			llm.calls,
		)
	}

	if len(llm.messages) != 1 {
		t.Fatalf(
			"expected one LLM request, got %d",
			len(llm.messages),
		)
	}

	messages := llm.messages[0]

	if len(messages) != 1 {
		t.Fatalf(
			"expected one message, got %d",
			len(messages),
		)
	}

	if messages[0].Role != "user" {
		t.Fatalf(
			"expected user role, got %q",
			messages[0].Role,
		)
	}

	if messages[0].Content != "hello" {
		t.Fatalf(
			"expected hello, got %q",
			messages[0].Content,
		)
	}
}
func TestConversationEngineStoresLLMResponse(t *testing.T) {
	llm := &testLLM{
		response: neurocall.Message{
			Role:    "assistant",
			Content: "I am fine.",
		},
	}

	engine := NewConversationEngine(
		NewLLMEngine(llm),
		nil,
	)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "call-1",
			Transcript: neurocall.Transcript{
				Text: "How are you?",
			},
		},
	)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	conversation, err := engine.GetOrCreate("call-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	messages := conversation.Messages()

	if len(messages) != 2 {
		t.Fatalf(
			"expected 2 messages, got %d",
			len(messages),
		)
	}

	if messages[0].Role != "user" {
		t.Fatalf("expected first message to be user")
	}

	if messages[0].Content != "How are you?" {
		t.Fatalf(
			"unexpected user message: %q",
			messages[0].Content,
		)
	}

	if messages[1].Role != "assistant" {
		t.Fatalf("expected second message to be assistant")
	}

	if messages[1].Content != "I am fine." {
		t.Fatalf(
			"unexpected assistant message: %q",
			messages[1].Content,
		)
	}
}
func TestConversationEngineLLMError(t *testing.T) {
	expectedErr := errors.New("llm unavailable")

	llm := &testLLM{
		err: expectedErr,
	}

	engine := NewConversationEngine(
		NewLLMEngine(llm),
		nil,
	)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "call-1",
			Transcript: neurocall.Transcript{
				Text: "hello",
			},
		},
	)

	if !errors.Is(err, expectedErr) {
		t.Fatalf(
			"expected LLM error, got %v",
			err,
		)
	}

	conversation, err := engine.GetOrCreate("call-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	messages := conversation.Messages()

	if len(messages) != 1 {
		t.Fatalf(
			"expected only user message after LLM failure, got %d",
			len(messages),
		)
	}
}
func TestConversationEngineEmptyTranscript(t *testing.T) {
	llm := &testLLM{
		response: neurocall.Message{
			Role:    "assistant",
			Content: "should not happen",
		},
	}

	engine := NewConversationEngine(
		NewLLMEngine(llm),
		nil,
	)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "call-1",
			Transcript: neurocall.Transcript{
				Text: "   ",
			},
		},
	)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if llm.calls != 0 {
		t.Fatalf(
			"LLM should not be called for empty transcript, got %d calls",
			llm.calls,
		)
	}
}
func TestTTSEngineSynthesize(t *testing.T) {
	fake := &testTTS{
		audio: neurocall.AudioStreamData{
			Format: neurocall.AudioFormat{
				SampleRate: 8000,
				Channels:   1,
			},
			Data: []byte{1, 2, 3, 4},
		},
	}

	engine := NewTTSEngine(fake)

	audio, err := engine.Synthesize(
		context.Background(),
		"hello",
	)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if fake.calls != 1 {
		t.Fatalf(
			"expected 1 TTS call, got %d",
			fake.calls,
		)
	}

	if len(fake.texts) != 1 || fake.texts[0] != "hello" {
		t.Fatalf(
			"unexpected TTS text: %#v",
			fake.texts,
		)
	}

	if len(audio.Data) != 4 {
		t.Fatalf(
			"expected 4 audio bytes, got %d",
			len(audio.Data),
		)
	}

	if audio.Format.SampleRate != 8000 {
		t.Fatalf(
			"expected sample rate 8000, got %d",
			audio.Format.SampleRate,
		)
	}
}
func TestTTSEngineSynthesizeWithoutTTS(t *testing.T) {
	engine := NewTTSEngine(nil)

	_, err := engine.Synthesize(
		context.Background(),
		"hello",
	)

	if err == nil {
		t.Fatal("expected error when TTS is not configured")
	}
}
func TestTTSEngineSynthesizeEmptyText(t *testing.T) {
	fake := &testTTS{}

	engine := NewTTSEngine(fake)

	_, err := engine.Synthesize(
		context.Background(),
		"   ",
	)

	if err == nil {
		t.Fatal("expected error for empty text")
	}

	if fake.calls != 0 {
		t.Fatalf(
			"TTS should not be called, got %d calls",
			fake.calls,
		)
	}
}
func TestTTSEngineSynthesizePropagatesError(t *testing.T) {
	expectedErr := errors.New("tts provider unavailable")

	fake := &testTTS{
		err: expectedErr,
	}

	engine := NewTTSEngine(fake)

	_, err := engine.Synthesize(
		context.Background(),
		"hello",
	)

	if !errors.Is(err, expectedErr) {
		t.Fatalf(
			"expected TTS error %v, got %v",
			expectedErr,
			err,
		)
	}

	if fake.calls != 1 {
		t.Fatalf(
			"expected 1 TTS call, got %d",
			fake.calls,
		)
	}
}
func TestTTSEngineSetTTS(t *testing.T) {
	first := &testTTS{
		audio: neurocall.AudioStreamData{
			Data: []byte{1},
		},
	}

	second := &testTTS{
		audio: neurocall.AudioStreamData{
			Data: []byte{2},
		},
	}

	engine := NewTTSEngine(first)

	_, err := engine.Synthesize(
		context.Background(),
		"first",
	)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	engine.SetTTS(second)

	audio, err := engine.Synthesize(
		context.Background(),
		"second",
	)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if first.calls != 1 {
		t.Fatalf(
			"expected first TTS to be called once, got %d",
			first.calls,
		)
	}

	if second.calls != 1 {
		t.Fatalf(
			"expected second TTS to be called once, got %d",
			second.calls,
		)
	}

	if len(audio.Data) != 1 || audio.Data[0] != 2 {
		t.Fatalf("unexpected audio from second TTS")
	}
}
func TestNewTTSPlayback(t *testing.T) {
	engine := NewTTSEngine(&testTTS{
		audio: neurocall.AudioStreamData{
			Data: []byte{1, 2, 3},
		},
	})

	playback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if playback == nil {
		t.Fatal("expected playback, got nil")
	}

	if playback.engine != engine {
		t.Fatal("playback should reference provided engine")
	}

	if playback.ctx == nil {
		t.Fatal("playback context is nil")
	}

	if playback.cancel == nil {
		t.Fatal("playback cancel function is nil")
	}
}
func TestNewTTSPlaybackNilEngine(t *testing.T) {
	playback, err := NewTTSPlayback(nil)

	if err == nil {
		t.Fatal("expected error for nil engine")
	}

	if playback != nil {
		t.Fatal("expected nil playback")
	}
}
func TestTTSPlaybackSynthesize(t *testing.T) {
	fake := &testTTS{
		audio: neurocall.AudioStreamData{
			Format: neurocall.AudioFormat{
				SampleRate: 8000,
				Channels:   1,
			},
			Data: []byte{10, 20, 30},
		},
	}

	engine := NewTTSEngine(fake)

	playback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	audio, err := playback.Synthesize(
		context.Background(),
		"hello",
	)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if fake.calls != 1 {
		t.Fatalf(
			"expected 1 TTS call, got %d",
			fake.calls,
		)
	}

	if fake.texts[0] != "hello" {
		t.Fatalf(
			"expected text %q, got %q",
			"hello",
			fake.texts[0],
		)
	}

	if len(audio.Data) != 3 {
		t.Fatalf(
			"expected 3 audio bytes, got %d",
			len(audio.Data),
		)
	}
}
func TestTTSPlaybackClose(t *testing.T) {
	engine := NewTTSEngine(&testTTS{})

	playback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	playbackCtx := playback.ctx

	if err := playback.Close(); err != nil {
		t.Fatalf("unexpected close error: %v", err)
	}

	select {
	case <-playbackCtx.Done():
	default:
		t.Fatal("playback context should be canceled")
	}
}
func TestTTSPlaybackCloseTwice(t *testing.T) {
	engine := NewTTSEngine(&testTTS{})

	playback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := playback.Close(); err != nil {
		t.Fatalf("first close failed: %v", err)
	}

	if err := playback.Close(); err != nil {
		t.Fatalf("second close failed: %v", err)
	}
}
func TestTTSPlaybackNilClose(t *testing.T) {
	var playback *TTSPlayback

	if err := playback.Close(); err != nil {
		t.Fatalf(
			"nil playback Close should not fail: %v",
			err,
		)
	}
}
func TestTTSPlaybackSynthesizeEmptyText(t *testing.T) {
	fake := &testTTS{
		audio: neurocall.AudioStreamData{
			Data: []byte{1},
		},
	}

	engine := NewTTSEngine(fake)

	playback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = playback.Synthesize(
		context.Background(),
		"   ",
	)

	if err == nil {
		t.Fatal("expected error for empty text")
	}

	if fake.calls != 0 {
		t.Fatalf(
			"TTS should not be called for empty text, got %d calls",
			fake.calls,
		)
	}
}
func TestTTSPlaybackSynthesizeError(t *testing.T) {
	expectedErr := errors.New("tts failed")

	fake := &testTTS{
		err: expectedErr,
	}

	engine := NewTTSEngine(fake)

	playback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = playback.Synthesize(
		context.Background(),
		"hello",
	)

	if !errors.Is(err, expectedErr) {
		t.Fatalf(
			"expected error %v, got %v",
			expectedErr,
			err,
		)
	}
}
func TestTTSPlaybackNilSynthesize(t *testing.T) {
	var playback *TTSPlayback

	_, err := playback.Synthesize(
		context.Background(),
		"hello",
	)

	if err == nil {
		t.Fatal("expected error for nil playback")
	}
}
func TestSIPCallSpeakWithoutTTS(t *testing.T) {
	call := &SIPCall{}

	err := call.Speak(context.Background(), "hello")

	if err == nil {
		t.Fatal("expected error")
	}

	if err.Error() != "TTS is not configured" {
		t.Fatalf("unexpected error: %v", err)
	}
}
func TestSIPCallSpeakWithoutRTP(t *testing.T) {
	call := &SIPCall{
		TTS: &TTSPlayback{},
	}

	err := call.Speak(context.Background(), "hello")

	if err == nil {
		t.Fatal("expected error")
	}
}
func TestSIPCallSpeakWithoutPipeline(t *testing.T) {
	call := &SIPCall{
		TTS: &TTSPlayback{},
		RTP: &RTPSession{},
	}

	err := call.Speak(context.Background(), "hello")

	if err == nil {
		t.Fatal("expected error")
	}
}
func TestSIPCallSpeakTTSError(t *testing.T) {
	expectedErr := errors.New("tts failed")

	fakeTTS := &testTTSFail{
		err: expectedErr,
	}

	engine := NewTTSEngine(fakeTTS)

	ttsPlayback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatal(err)
	}
	defer ttsPlayback.Close()

	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		50000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rtp.Close()

	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})

	call := &SIPCall{
		TTS:      ttsPlayback,
		RTP:      rtp,
		Pipeline: pipeline,
		Codec:    codec,
	}

	err = call.Speak(context.Background(), "hello")

	if !errors.Is(err, expectedErr) {
		t.Fatalf(
			"expected TTS error %v, got %v",
			expectedErr,
			err,
		)
	}

	if fakeTTS.calls != 1 {
		t.Fatalf(
			"expected 1 TTS call, got %d",
			fakeTTS.calls,
		)
	}
}
func TestSIPCallSpeakSuccess(t *testing.T) {
	fakeTTS := &testTTSFail{
		data: neurocall.AudioStreamData{
			Format: neurocall.AudioFormat{
				SampleRate: 8000,
				Channels:   1,
			},
			Data: make([]byte, 320),
		},
	}

	engine := NewTTSEngine(fakeTTS)

	ttsPlayback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatal(err)
	}
	defer ttsPlayback.Close()

	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		50000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rtp.Close()

	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})

	pipeline.SetEncoder(PCMU{})
	pipeline.SetDecoder(PCMU{})

	call := &SIPCall{
		TTS:      ttsPlayback,
		RTP:      rtp,
		Pipeline: pipeline,
		Codec:    codec,
	}

	err = call.Speak(
		context.Background(),
		"hello",
	)

	if err != nil {
		t.Fatalf("expected Speak to succeed, got %v", err)
	}

	if fakeTTS.calls != 1 {
		t.Fatalf(
			"expected 1 TTS call, got %d",
			fakeTTS.calls,
		)
	}
}
func TestSIPCallSpeakInvalidTTSAudio(t *testing.T) {
	fakeTTS := &testTTSFail{
		data: neurocall.AudioStreamData{
			Format: neurocall.AudioFormat{
				SampleRate: 8000,
				Channels:   1,
			},
			Data: nil,
		},
	}

	engine := NewTTSEngine(fakeTTS)

	ttsPlayback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatal(err)
	}
	defer ttsPlayback.Close()

	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		50000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rtp.Close()

	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})

	pipeline.SetEncoder(PCMU{})
	pipeline.SetDecoder(PCMU{})

	call := &SIPCall{
		TTS:      ttsPlayback,
		RTP:      rtp,
		Pipeline: pipeline,
		Codec:    codec,
	}

	err = call.Speak(
		context.Background(),
		"hello",
	)

	if !errors.Is(err, neurocall.ErrInvalidAudio) {
		t.Fatalf(
			"expected ErrInvalidAudio, got %v",
			err,
		)
	}

	if fakeTTS.calls != 1 {
		t.Fatalf(
			"expected 1 TTS call, got %d",
			fakeTTS.calls,
		)
	}
}
func TestConversationToVoiceIntegration(t *testing.T) {
	bus := NewEventBus()
	llm := &integrationTestLLM{
		response: neurocall.Message{
			Role:    "assistant",
			Content: "hello from AI",
		},
	}
	llmEngine := NewLLMEngine(llm)
	conversationEngine := NewConversationEngine(
		llmEngine,
		bus,
	)
	tts := &integrationTestTTS{
		data: neurocall.AudioStreamData{
			Format: neurocall.AudioFormat{
				SampleRate: 8000,
				Channels:   1,
			},
			Data: PCM16ToBytes([]int16{
				100,
				200,
				300,
				400,
			}),
		},
	}
	ttsEngine := NewTTSEngine(tts)
	voiceEngine := NewVoiceResponseEngine(
		ttsEngine,
		bus,
	)
	call := &SIPCall{
		CallID: "integration-call",
	}
	if err := voiceEngine.AddCall(call); err != nil {
		t.Fatal(err)
	}
	err := conversationEngine.HandleTranscript(
		TranscriptEvent{
			CallID: "integration-call",
			Transcript: neurocall.Transcript{
				Text: "hello",
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if llm.calls != 1 {
		t.Fatalf(
			"expected LLM to be called once, got %d",
			llm.calls,
		)
	}
	conversation, err := conversationEngine.GetOrCreate(
		"integration-call",
	)
	if err != nil {
		t.Fatal(err)
	}
	messages := conversation.Messages()
	if len(messages) != 2 {
		t.Fatalf(
			"expected 2 conversation messages, got %d",
			len(messages),
		)
	}
	if messages[0].Role != "user" ||
		messages[0].Content != "hello" {
		t.Fatalf(
			"unexpected user message: %+v",
			messages[0],
		)
	}
	if messages[1].Role != "assistant" ||
		messages[1].Content != "hello from AI" {
		t.Fatalf(
			"unexpected assistant message: %+v",
			messages[1],
		)
	}
	err = voiceEngine.HandleResponse(
		LLMResponseEvent{
			CallID:  "integration-call",
			Message: llm.response,
		},
	)
	if err == nil {
		t.Fatal(
			"expected Speak to fail because RTP/Pipeline are not configured",
		)
	}
	if tts.calls != 0 {
		t.Fatalf(
			"TTS should not be called before RTP/Pipeline validation, got %d",
			tts.calls,
		)
	}
}
func TestRTPSessionWritePCMRealtime(t *testing.T) {
	receiver, err := net.ListenUDP(
		"udp",
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: 0,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer receiver.Close()

	receiverPort := receiver.LocalAddr().(*net.UDPAddr).Port

	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		receiverPort,
		neurocall.Codec{
			Name:        "PCMU",
			PayloadType: 0,
			ClockRate:   8000,
			Channels:    1,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rtp.Close()

	pcm := make([]int16, 320)

	err = rtp.WritePCMRealtime(
		context.Background(),
		PCMU{},
		pcm,
	)
	if err != nil {
		t.Fatal(err)
	}

	_ = receiver.SetReadDeadline(
		time.Now().Add(time.Second),
	)

	received := 0

	for received < 2 {
		buf := make([]byte, 2048)

		n, _, err := receiver.ReadFromUDP(buf)
		if err != nil {
			t.Fatal(err)
		}

		packet := &RTPPacket{}

		if err := packet.Unmarshal(buf[:n]); err != nil {
			t.Fatalf(
				"invalid RTP packet: %v",
				err,
			)
		}

		if packet.Header.Version != 2 {
			t.Fatalf(
				"expected RTP version 2, got %d",
				packet.Header.Version,
			)
		}

		if packet.Header.PayloadType != 0 {
			t.Fatalf(
				"expected payload type 0, got %d",
				packet.Header.PayloadType,
			)
		}

		if len(packet.Payload) != 160 {
			t.Fatalf(
				"expected 160-byte payload, got %d",
				len(packet.Payload),
			)
		}

		received++
	}

	if received != 2 {
		t.Fatalf(
			"expected 2 RTP packets, got %d",
			received,
		)
	}
}
func TestRTPSessionWritePCMRealtimeContextCancel(t *testing.T) {
	receiver, err := net.ListenUDP(
		"udp",
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: 0,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer receiver.Close()

	receiverPort := receiver.LocalAddr().(*net.UDPAddr).Port

	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		receiverPort,
		neurocall.Codec{
			Name:        "PCMU",
			PayloadType: 0,
			ClockRate:   8000,
			Channels:    1,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rtp.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pcm := make([]int16, 320)

	done := make(chan error, 1)

	go func() {
		done <- rtp.WritePCMRealtime(
			ctx,
			PCMU{},
			pcm,
		)
	}()

	time.Sleep(5 * time.Millisecond)

	cancel()

	err = <-done

	if !errors.Is(err, context.Canceled) {
		t.Fatalf(
			"expected context.Canceled, got %v",
			err,
		)
	}
}
func TestRTPJitterBufferReorder(t *testing.T) {
	buffer := NewRTPJitterBuffer(
		10,
		100*time.Millisecond,
	)

	p100 := &RTPPacket{}
	p100.Header.SequenceNumber = 100

	p101 := &RTPPacket{}
	p101.Header.SequenceNumber = 101

	p102 := &RTPPacket{}
	p102.Header.SequenceNumber = 102

	if err := buffer.Push(p102); err != nil {
		t.Fatalf("Push 102 failed: %v", err)
	}

	if err := buffer.Push(p100); err != nil {
		t.Fatalf("Push 100 failed: %v", err)
	}

	if err := buffer.Push(p101); err != nil {
		t.Fatalf("Push 101 failed: %v", err)
	}

	got, ok := buffer.Pop()

	if !ok || got == nil {
		t.Fatal("expected packet 100")
	}

	if got.Header.SequenceNumber != 100 {
		t.Fatalf(
			"expected 100, got %d",
			got.Header.SequenceNumber,
		)
	}

	got, ok = buffer.Pop()

	if !ok || got == nil {
		t.Fatal("expected packet 101")
	}

	if got.Header.SequenceNumber != 101 {
		t.Fatalf(
			"expected 101, got %d",
			got.Header.SequenceNumber,
		)
	}

	got, ok = buffer.Pop()

	if !ok || got == nil {
		t.Fatal("expected packet 102")
	}

	if got.Header.SequenceNumber != 102 {
		t.Fatalf(
			"expected 102, got %d",
			got.Header.SequenceNumber,
		)
	}
}
func TestSIPCallSpeakEndToEndRTP(t *testing.T) {
	receiver, err := net.ListenUDP(
		"udp",
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: 0,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer receiver.Close()

	receiverPort := receiver.LocalAddr().(*net.UDPAddr).Port

	fakeTTS := &testTTSFail{
		data: neurocall.AudioStreamData{
			Format: neurocall.AudioFormat{
				SampleRate: 8000,
				Channels:   1,
			},
			Data: PCM16ToBytes(make([]int16, 160)),
		},
	}

	ttsEngine := NewTTSEngine(fakeTTS)

	ttsPlayback, err := NewTTSPlayback(ttsEngine)
	if err != nil {
		t.Fatal(err)
	}
	defer ttsPlayback.Close()
	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		receiverPort,
		neurocall.Codec{
			Name:        "PCMU",
			PayloadType: 0,
			ClockRate:   8000,
			Channels:    1,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rtp.Close()
	pipeline := NewAudioPipeline(
		AudioConfig{
			BufferSize: 10,
		},
	)
	defer pipeline.Close()

	pipeline.SetEncoder(PCMU{})
	pipeline.SetDecoder(PCMU{})

	call := &SIPCall{
		CallID:   "e2e-call",
		TTS:      ttsPlayback,
		RTP:      rtp,
		Pipeline: pipeline,
		Codec: neurocall.Codec{
			Name:        "PCMU",
			PayloadType: 0,
			ClockRate:   8000,
			Channels:    1,
		},
	}

	err = call.Speak(
		context.Background(),
		"hello",
	)
	if err != nil {
		t.Fatal(err)
	}

	if fakeTTS.calls != 1 {
		t.Fatalf(
			"expected TTS to be called once, got %d",
			fakeTTS.calls,
		)
	}

	_ = receiver.SetReadDeadline(
		time.Now().Add(time.Second),
	)

	buf := make([]byte, 2048)

	n, _, err := receiver.ReadFromUDP(buf)
	if err != nil {
		t.Fatal(err)
	}

	if n <= 12 {
		t.Fatalf(
			"expected RTP packet with payload, got %d bytes",
			n,
		)
	}

	packet := &RTPPacket{}

	if err := packet.Unmarshal(buf[:n]); err != nil {
		t.Fatalf(
			"failed to decode RTP packet: %v",
			err,
		)
	}

	if packet.Header.Version != 2 {
		t.Fatalf(
			"expected RTP version 2, got %d",
			packet.Header.Version,
		)
	}

	if packet.Header.PayloadType != 0 {
		t.Fatalf(
			"expected PCMU payload type 0, got %d",
			packet.Header.PayloadType,
		)
	}

	if len(packet.Payload) != 160 {
		t.Fatalf(
			"expected 160-byte PCMU payload, got %d",
			len(packet.Payload),
		)
	}
}
func TestRTPSessionReadValidPacket(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	session.mu.RLock()
	localPort := session.conn.LocalAddr().(*net.UDPAddr).Port
	session.mu.RUnlock()

	packet := &RTPPacket{
		Header: RTPHeader{
			Version:        2,
			PayloadType:    codec.PayloadType,
			SequenceNumber: 1,
			Timestamp:      160,
			SSRC:           1234,
		},
		Payload: []byte{0x7f, 0x7f, 0x7f},
	}

	data, err := packet.Marshal()
	if err != nil {
		t.Fatal(err)
	}

	conn, err := net.DialUDP(
		"udp",
		nil,
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: localPort,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	if _, err := conn.Write(data); err != nil {
		t.Fatal(err)
	}

	got, err := session.Read()
	if err != nil {
		t.Fatal(err)
	}

	if got == nil {
		t.Fatal("expected RTP packet")
	}

	if got.Header.SequenceNumber != 1 {
		t.Fatalf(
			"expected sequence 1, got %d",
			got.Header.SequenceNumber,
		)
	}

	if !bytes.Equal(got.Payload, packet.Payload) {
		t.Fatalf("payload mismatch")
	}
}
func TestRTPSessionReadInvalidPacket(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	session.mu.RLock()
	localPort := session.conn.LocalAddr().(*net.UDPAddr).Port
	session.mu.RUnlock()

	conn, err := net.DialUDP(
		"udp",
		nil,
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: localPort,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	if _, err := conn.Write([]byte{0x01, 0x02, 0x03}); err != nil {
		t.Fatal(err)
	}

	_, err = session.Read()
	if err == nil {
		t.Fatal("expected invalid RTP packet error")
	}

	stats := session.Stats()

	if stats.InvalidPackets != 1 {
		t.Fatalf(
			"expected 1 invalid packet, got %d",
			stats.InvalidPackets,
		)
	}
}
func TestRTPSessionReadClosed(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	_, err = session.Read()
	if err == nil {
		t.Fatal("expected error after session close")
	}
}
func TestRTPSessionReadPCM(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	session.mu.RLock()
	localPort := session.conn.LocalAddr().(*net.UDPAddr).Port
	session.mu.RUnlock()

	conn, err := net.DialUDP(
		"udp",
		nil,
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: localPort,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	pcm := make([]int16, 160)

	for i := range pcm {
		pcm[i] = 1000
	}

	encoded := (PCMU{}).Encode(pcm)

	packet := &RTPPacket{
		Header: RTPHeader{
			Version:        2,
			PayloadType:    codec.PayloadType,
			SequenceNumber: 1,
			Timestamp:      0,
			SSRC:           1234,
		},
		Payload: encoded,
	}

	data, err := packet.Marshal()
	if err != nil {
		t.Fatal(err)
	}

	if _, err := conn.Write(data); err != nil {
		t.Fatal(err)
	}

	got, err := session.ReadPCM(PCMU{})
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != len(pcm) {
		t.Fatalf(
			"expected %d PCM samples, got %d",
			len(pcm),
			len(got),
		)
	}
}
func TestRTPSessionReadPCMNilDecoder(t *testing.T) {
	session := &RTPSession{}

	_, err := session.ReadPCM(nil)

	if err == nil {
		t.Fatal("expected decoder nil error")
	}
}
func TestRTPSessionReadPCMClosed(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}

	session.Close()

	_, err = session.ReadPCM(PCMU{})
	if err == nil {
		t.Fatal("expected closed session error")
	}
}
func TestRTPSessionReadPCMWrongPayloadType(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	session.mu.RLock()
	localPort := session.conn.LocalAddr().(*net.UDPAddr).Port
	session.mu.RUnlock()

	conn, err := net.DialUDP(
		"udp",
		nil,
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: localPort,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	packet := &RTPPacket{
		Header: RTPHeader{
			Version:        2,
			PayloadType:    8,
			SequenceNumber: 1,
			Timestamp:      0,
			SSRC:           1234,
		},
		Payload: []byte{0x7f},
	}

	data, err := packet.Marshal()
	if err != nil {
		t.Fatal(err)
	}

	if _, err := conn.Write(data); err != nil {
		t.Fatal(err)
	}

	_, err = session.ReadPCM(PCMU{})
	if err == nil {
		t.Fatal("expected payload type error")
	}
}
func TestRTPSessionReadOrdered(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	session.mu.RLock()
	localPort := session.conn.LocalAddr().(*net.UDPAddr).Port
	session.mu.RUnlock()

	conn, err := net.DialUDP(
		"udp",
		nil,
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: localPort,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	for i := uint16(1); i <= 3; i++ {
		packet := &RTPPacket{
			Header: RTPHeader{
				Version:        2,
				PayloadType:    codec.PayloadType,
				SequenceNumber: i,
				Timestamp:      uint32(i * 160),
				SSRC:           1234,
			},
			Payload: []byte{byte(i)},
		}

		data, err := packet.Marshal()
		if err != nil {
			t.Fatal(err)
		}

		if _, err := conn.Write(data); err != nil {
			t.Fatal(err)
		}
	}

	for expected := uint16(1); expected <= 3; expected++ {
		result, err := session.ReadOrdered()
		if err != nil {
			t.Fatal(err)
		}

		if result.Lost {
			t.Fatalf("unexpected packet loss at sequence %d", expected)
		}

		if result.Packet == nil {
			t.Fatal("expected RTP packet")
		}

		if result.Packet.Header.SequenceNumber != expected {
			t.Fatalf(
				"expected sequence %d, got %d",
				expected,
				result.Packet.Header.SequenceNumber,
			)
		}
	}
}
func TestRTPSessionReadOrderedLoss(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	session.mu.RLock()
	localPort := session.conn.LocalAddr().(*net.UDPAddr).Port
	session.mu.RUnlock()

	conn, err := net.DialUDP(
		"udp",
		nil,
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: localPort,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	for _, seq := range []uint16{1, 3} {
		packet := &RTPPacket{
			Header: RTPHeader{
				Version:        2,
				PayloadType:    codec.PayloadType,
				SequenceNumber: seq,
				Timestamp:      uint32(seq * 160),
				SSRC:           1234,
			},
			Payload: []byte{byte(seq)},
		}

		data, err := packet.Marshal()
		if err != nil {
			t.Fatal(err)
		}

		if _, err := conn.Write(data); err != nil {
			t.Fatal(err)
		}
	}

	result, err := session.ReadOrdered()
	if err != nil {
		t.Fatal(err)
	}

	if result.Lost {
		t.Fatal("first packet should not be lost")
	}

	if result.Packet == nil {
		t.Fatal("expected first packet")
	}

	if result.Packet.Header.SequenceNumber != 1 {
		t.Fatalf(
			"expected sequence 1, got %d",
			result.Packet.Header.SequenceNumber,
		)
	}

	result, err = session.ReadOrdered()
	if err != nil {
		t.Fatal(err)
	}

	if !result.Lost {
		t.Fatal("expected second packet to be reported as lost")
	}
}
func TestSIPCallSpeakAfterClose(t *testing.T) {
	call := &SIPCall{}

	if err := call.CloseResources(); err != nil {
		t.Fatal(err)
	}

	err := call.Speak(
		context.Background(),
		"hello",
	)

	if !errors.Is(err, neurocall.ErrCallClosed) {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestSIPCallSpeakContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	fakeTTS := &testTTSFail{
		data: neurocall.AudioStreamData{
			Data: []byte{0, 1, 2, 3},
		},
	}

	engine := NewTTSEngine(fakeTTS)

	ttsPlayback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatal(err)
	}

	call := &SIPCall{
		TTS: ttsPlayback,
	}

	err = call.Speak(ctx, "hello")

	if err == nil {
		t.Fatal("expected error from canceled context")
	}
}
func TestSIPCallSpeakInvalidTTSSampleRate(t *testing.T) {
	fakeTTS := &testTTSFail{
		data: neurocall.AudioStreamData{
			Data: []byte{0, 1, 2, 3},
			Format: neurocall.AudioFormat{
				SampleRate: 0,
			},
		},
	}

	engine := NewTTSEngine(fakeTTS)

	ttsPlayback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatal(err)
	}

	call := &SIPCall{
		TTS: ttsPlayback,
	}

	err = call.Speak(
		context.Background(),
		"hello",
	)

	if err == nil {
		t.Fatal("expected error for invalid TTS sample rate")
	}
}
func TestSIPCallSpeakAfterCloseDoesNotCallTTS(t *testing.T) {
	fakeTTS := &testTTSFail{
		data: neurocall.AudioStreamData{
			Data: []byte{0, 1, 2, 3},
		},
	}

	engine := NewTTSEngine(fakeTTS)

	ttsPlayback, err := NewTTSPlayback(engine)
	if err != nil {
		t.Fatal(err)
	}

	call := &SIPCall{
		TTS: ttsPlayback,
	}

	if err := call.CloseResources(); err != nil {
		t.Fatal(err)
	}

	err = call.Speak(
		context.Background(),
		"hello",
	)

	if !errors.Is(err, neurocall.ErrCallClosed) {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}

	if fakeTTS.calls != 0 {
		t.Fatalf(
			"expected TTS not to be called, got %d calls",
			fakeTTS.calls,
		)
	}
}
func TestRTPSessionWritePCMRealtimeContextCancelDuringPlayback(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9999,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	ctx, cancel := context.WithCancel(context.Background())

	pcm := make([]int16, 8000)

	done := make(chan error, 1)

	go func() {
		done <- session.WritePCMRealtime(
			ctx,
			PCMU{},
			pcm,
		)
	}()

	time.Sleep(30 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf(
				"expected context.Canceled, got %v",
				err,
			)
		}

	case <-time.After(time.Second):
		t.Fatal("WritePCMRealtime did not stop after context cancellation")
	}
}
func TestRTPSessionWritePCMRealtimeSequenceAndTimestamp(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9999,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	session.writeInterval = 0

	pcm := make([]int16, 320)

	err = session.WritePCMRealtime(
		context.Background(),
		PCMU{},
		pcm,
	)
	if err != nil {
		t.Fatal(err)
	}

	session.mu.RLock()
	sequence := session.sequence
	timestamp := session.timestamp
	session.mu.RUnlock()

	if sequence != 2 {
		t.Fatalf(
			"expected sequence 2 after 2 frames, got %d",
			sequence,
		)
	}

	if timestamp != 320 {
		t.Fatalf(
			"expected timestamp 320 after 320 samples, got %d",
			timestamp,
		)
	}
}
func TestRTPSessionTimestampWrap(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9999,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	session.mu.Lock()
	session.timestamp = ^uint32(0) - 79
	session.mu.Unlock()

	session.AdvanceTimestamp(160)

	session.mu.RLock()
	timestamp := session.timestamp
	session.mu.RUnlock()

	if timestamp != 80 {
		t.Fatalf(
			"expected timestamp 80 after wrap-around, got %d",
			timestamp,
		)
	}
}
func TestRTPSessionSequenceWrap(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9999,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	session.mu.Lock()
	session.sequence = 65535
	session.mu.Unlock()

	pcm := make([]int16, 160)

	if err := session.WritePCM(PCMU{}, pcm); err != nil {
		t.Fatal(err)
	}

	session.mu.RLock()
	sequence := session.sequence
	session.mu.RUnlock()

	if sequence != 0 {
		t.Fatalf(
			"expected sequence to wrap to 0, got %d",
			sequence,
		)
	}

	if err := session.WritePCM(PCMU{}, pcm); err != nil {
		t.Fatal(err)
	}

	session.mu.RLock()
	sequence = session.sequence
	session.mu.RUnlock()

	if sequence != 1 {
		t.Fatalf(
			"expected sequence 1 after wrap-around, got %d",
			sequence,
		)
	}
}
func TestRTPSessionSSRCConsistency(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9999,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	session.mu.RLock()
	expectedSSRC := session.ssrc
	session.mu.RUnlock()

	if expectedSSRC == 0 {
		t.Fatal("expected non-zero SSRC")
	}

	pcm := make([]int16, 160)

	if err := session.WritePCM(PCMU{}, pcm); err != nil {
		t.Fatal(err)
	}

	session.mu.RLock()
	actualSSRC := session.ssrc
	session.mu.RUnlock()

	if actualSSRC != expectedSSRC {
		t.Fatalf(
			"SSRC changed: expected %d, got %d",
			expectedSSRC,
			actualSSRC,
		)
	}
}
func TestRTPSessionPayloadTypeConsistency(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9999,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	if session.codec.PayloadType != codec.PayloadType {
		t.Fatalf(
			"expected payload type %d, got %d",
			codec.PayloadType,
			session.codec.PayloadType,
		)
	}

	pcm := make([]int16, 160)

	if err := session.WritePCM(PCMU{}, pcm); err != nil {
		t.Fatal(err)
	}

	if session.codec.PayloadType != 0 {
		t.Fatalf(
			"expected PCMU payload type 0, got %d",
			session.codec.PayloadType,
		)
	}
}
func TestVoiceResponseEngineHandleResponse(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9999,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rtp.Close()

	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})

	pipeline.SetEncoder(PCMU{})
	pipeline.SetDecoder(PCMU{})

	fakeTTS := &testTTSFail{
		data: neurocall.AudioStreamData{
			Data: []byte{0, 1, 2, 3},
			Format: neurocall.AudioFormat{
				SampleRate: 8000,
			},
		},
	}

	ttsEngine := NewTTSEngine(fakeTTS)

	callTTS, err := NewTTSPlayback(ttsEngine)
	if err != nil {
		t.Fatal(err)
	}

	call := &SIPCall{
		CallID:   "call-1",
		TTS:      callTTS,
		RTP:      rtp,
		Pipeline: pipeline,
		Codec:    codec,
	}

	engine := NewVoiceResponseEngine(
		ttsEngine,
		nil,
	)

	if err := engine.AddCall(call); err != nil {
		t.Fatal(err)
	}

	err = engine.HandleResponse(
		LLMResponseEvent{
			CallID: "call-1",
			Message: neurocall.Message{
				Role:    "assistant",
				Content: "hello",
			},
		},
	)

	if err != nil {
		t.Fatal(err)
	}

	if fakeTTS.calls != 1 {
		t.Fatalf(
			"expected TTS to be called once, got %d",
			fakeTTS.calls,
		)
	}
}
func TestVoiceResponseEngineHandleResponseCallNotFound22(t *testing.T) {
	fakeTTS := &testTTSFail{
		data: neurocall.AudioStreamData{
			Data: []byte{0, 1, 2, 3},
			Format: neurocall.AudioFormat{
				SampleRate: 8000,
			},
		},
	}

	ttsEngine := NewTTSEngine(fakeTTS)

	engine := NewVoiceResponseEngine(
		ttsEngine,
		nil,
	)

	err := engine.HandleResponse(
		LLMResponseEvent{
			CallID: "unknown-call",
			Message: neurocall.Message{
				Role:    "assistant",
				Content: "hello",
			},
		},
	)

	if !errors.Is(err, neurocall.ErrCallNotFound) {
		t.Fatalf(
			"expected ErrCallNotFound, got %v",
			err,
		)
	}

	if fakeTTS.calls != 0 {
		t.Fatalf(
			"expected TTS not to be called, got %d calls",
			fakeTTS.calls,
		)
	}
}
func TestVoiceResponseEngineHandleResponseClosedCall2(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9999,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}

	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})
	pipeline.SetEncoder(PCMU{})
	pipeline.SetDecoder(PCMU{})

	fakeTTS := &testTTSFail{
		data: neurocall.AudioStreamData{
			Data: []byte{0, 1, 2, 3},
			Format: neurocall.AudioFormat{
				SampleRate: 8000,
			},
		},
	}

	ttsEngine := NewTTSEngine(fakeTTS)

	callTTS, err := NewTTSPlayback(ttsEngine)
	if err != nil {
		rtp.Close()
		t.Fatal(err)
	}

	call := &SIPCall{
		CallID:   "call-closed",
		TTS:      callTTS,
		RTP:      rtp,
		Pipeline: pipeline,
		Codec:    codec,
	}

	engine := NewVoiceResponseEngine(
		ttsEngine,
		nil,
	)

	if err := engine.AddCall(call); err != nil {
		call.CloseResources()
		t.Fatal(err)
	}

	if err := call.CloseResources(); err != nil {
		t.Fatal(err)
	}

	err = engine.HandleResponse(
		LLMResponseEvent{
			CallID: "call-closed",
			Message: neurocall.Message{
				Role:    "assistant",
				Content: "hello",
			},
		},
	)

	if !errors.Is(err, neurocall.ErrCallClosed) {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}

	if fakeTTS.calls != 0 {
		t.Fatalf(
			"expected TTS not to be called, got %d calls",
			fakeTTS.calls,
		)
	}
}
func TestVoiceResponseEngineLLMResponseEvent(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9999,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rtp.Close()

	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})
	pipeline.SetEncoder(PCMU{})
	pipeline.SetDecoder(PCMU{})

	fakeTTS := &testTTSFail{
		data: neurocall.AudioStreamData{
			Data: []byte{0, 1, 2, 3},
			Format: neurocall.AudioFormat{
				SampleRate: 8000,
			},
		},
	}

	ttsEngine := NewTTSEngine(fakeTTS)

	callTTS, err := NewTTSPlayback(ttsEngine)
	if err != nil {
		t.Fatal(err)
	}

	call := &SIPCall{
		CallID:   "event-call",
		TTS:      callTTS,
		RTP:      rtp,
		Pipeline: pipeline,
		Codec:    codec,
	}

	bus := NewEventBus()

	engine := NewVoiceResponseEngine(
		ttsEngine,
		bus,
	)

	if err := engine.AddCall(call); err != nil {
		t.Fatal(err)
	}

	bus.Publish(Event{
		Name: EventLLMResponse,
		Data: LLMResponseEvent{
			CallID: "event-call",
			Message: neurocall.Message{
				Role:    "assistant",
				Content: "hello from event",
			},
		},
	})

	if fakeTTS.calls != 1 {
		t.Fatalf(
			"expected TTS to be called once, got %d",
			fakeTTS.calls,
		)
	}
}
func TestVoiceResponseEngineTTSErrorEvent(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9999,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rtp.Close()

	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})
	pipeline.SetEncoder(PCMU{})
	pipeline.SetDecoder(PCMU{})

	fakeTTS := &testTTSFail{
		err: errors.New("tts failed"),
	}

	ttsEngine := NewTTSEngine(fakeTTS)

	callTTS, err := NewTTSPlayback(ttsEngine)
	if err != nil {
		t.Fatal(err)
	}

	call := &SIPCall{
		CallID:   "tts-error-call",
		TTS:      callTTS,
		RTP:      rtp,
		Pipeline: pipeline,
		Codec:    codec,
	}

	bus := NewEventBus()

	engine := NewVoiceResponseEngine(
		ttsEngine,
		bus,
	)

	if err := engine.AddCall(call); err != nil {
		t.Fatal(err)
	}

	var received bool
	var receivedErr error
	var receivedCallID string

	bus.Subscribe(
		EventTTSResponseError,
		func(event Event) {
			data, ok := event.Data.(TTSErrorEvent)
			if !ok {
				return
			}

			received = true
			receivedErr = data.Err
			receivedCallID = data.CallID
		},
	)

	bus.Publish(Event{
		Name: EventLLMResponse,
		Data: LLMResponseEvent{
			CallID: "tts-error-call",
			Message: neurocall.Message{
				Role:    "assistant",
				Content: "hello",
			},
		},
	})

	if !received {
		t.Fatal("expected EventTTSResponseError")
	}

	if receivedCallID != "tts-error-call" {
		t.Fatalf(
			"expected call ID tts-error-call, got %s",
			receivedCallID,
		)
	}

	if receivedErr == nil {
		t.Fatal("expected TTS error")
	}

	if receivedErr.Error() != "tts failed" {
		t.Fatalf(
			"expected tts failed, got %v",
			receivedErr,
		)
	}
}
func TestConversationEnginePublishesLLMResponseEvent(t *testing.T) {
	bus := NewEventBus()

	fakeLLM := &testLLM{
		response: neurocall.Message{
			Role:    "assistant",
			Content: "hello from llm",
		},
	}

	llm := &LLMEngine{
		llm: fakeLLM,
	}

	engine := NewConversationEngine(
		llm,
		bus,
	)

	var received bool
	var receivedCallID string
	var receivedContent string

	bus.Subscribe(
		EventLLMResponse,
		func(event Event) {
			data, ok := event.Data.(LLMResponseEvent)
			if !ok {
				return
			}

			received = true
			receivedCallID = data.CallID
			receivedContent = data.Message.Content
		},
	)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "conversation-event-call",
			Transcript: neurocall.Transcript{
				Text: "hello",
			},
		},
	)

	if err != nil {
		t.Fatal(err)
	}

	if !received {
		t.Fatal("expected EventLLMResponse")
	}

	if receivedCallID != "conversation-event-call" {
		t.Fatalf(
			"expected call ID conversation-event-call, got %s",
			receivedCallID,
		)
	}

	if receivedContent != "hello from llm" {
		t.Fatalf(
			"expected hello from llm, got %s",
			receivedContent,
		)
	}
}
func TestConversationEnginePublishesLLMErrorEvent(t *testing.T) {
	bus := NewEventBus()

	fakeLLM := &testLLM{
		err: errors.New("llm failed"),
	}

	llm := &LLMEngine{
		llm: fakeLLM,
	}

	engine := NewConversationEngine(
		llm,
		bus,
	)

	var received bool
	var receivedErr error

	bus.Subscribe(
		EventLLMError,
		func(event Event) {
			err, ok := event.Data.(error)
			if !ok {
				return
			}

			received = true
			receivedErr = err
		},
	)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "llm-error-call",
			Transcript: neurocall.Transcript{
				Text: "hello",
			},
		},
	)

	if err == nil {
		t.Fatal("expected LLM error")
	}

	if err.Error() != "llm failed" {
		t.Fatalf(
			"expected llm failed, got %v",
			err,
		)
	}

	if !received {
		t.Fatal("expected EventLLMError")
	}

	if receivedErr == nil {
		t.Fatal("expected error in EventLLMError")
	}

	if receivedErr.Error() != "llm failed" {
		t.Fatalf(
			"expected event error llm failed, got %v",
			receivedErr,
		)
	}
}
func TestConversationEngineHandleTranscriptWhitespace2(t *testing.T) {
	fakeLLM := &testLLM{}

	llm := &LLMEngine{
		llm: fakeLLM,
	}

	engine := NewConversationEngine(llm, nil)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "test-call",
			Transcript: neurocall.Transcript{
				Text: "   \t\n   ",
			},
		},
	)

	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}

	conversation, err := engine.GetOrCreate("test-call")
	if err != nil {
		t.Fatal(err)
	}

	if len(conversation.Messages()) != 0 {
		t.Fatalf(
			"expected no messages, got %d",
			len(conversation.Messages()),
		)
	}
}
func TestConversationEngineGetOrCreateReturnsSameConversation(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	first, err := engine.GetOrCreate("call-123")
	if err != nil {
		t.Fatal(err)
	}

	second, err := engine.GetOrCreate("call-123")
	if err != nil {
		t.Fatal(err)
	}

	if first == nil || second == nil {
		t.Fatal("expected non-nil conversations")
	}

	if first != second {
		t.Fatal("expected same conversation instance")
	}
}
func TestConversationEngineGetOrCreateEmptyCallID2(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	conversation, err := engine.GetOrCreate("")

	if err == nil {
		t.Fatal("expected error for empty call ID")
	}

	if conversation != nil {
		t.Fatal("expected nil conversation")
	}
}
func TestConversationEngineSeparatesCalls(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	first, err := engine.GetOrCreate("call-1")
	if err != nil {
		t.Fatal(err)
	}

	second, err := engine.GetOrCreate("call-2")
	if err != nil {
		t.Fatal(err)
	}

	if first == nil || second == nil {
		t.Fatal("expected non-nil conversations")
	}

	if first == second {
		t.Fatal("expected different conversations for different call IDs")
	}

	first.AddMessage(neurocall.Message{
		Role:    "user",
		Content: "hello",
	})

	if len(first.Messages()) != 1 {
		t.Fatalf(
			"expected first conversation to contain 1 message, got %d",
			len(first.Messages()),
		)
	}

	if len(second.Messages()) != 0 {
		t.Fatalf(
			"expected second conversation to remain empty, got %d messages",
			len(second.Messages()),
		)
	}
}
func TestConversationEngineGetOrCreateEmptyCallIDJustengine(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	_, err := engine.GetOrCreate("")

	if err == nil {
		t.Fatal("expected error for empty call ID")
	}
}
func TestConversationEngineHandleMultipleTranscripts(t *testing.T) {
	fakeLLM := &testLLM{
		response: neurocall.Message{
			Role:    "assistant",
			Content: "response",
		},
	}

	llm := &LLMEngine{
		llm: fakeLLM,
	}

	engine := NewConversationEngine(llm, nil)

	for _, text := range []string{"hello", "how are you?"} {
		err := engine.HandleTranscript(
			TranscriptEvent{
				CallID: "multi-call",
				Transcript: neurocall.Transcript{
					Text: text,
				},
			},
		)

		if err != nil {
			t.Fatal(err)
		}
	}

	conversation, err := engine.GetOrCreate("multi-call")
	if err != nil {
		t.Fatal(err)
	}

	messages := conversation.Messages()

	if len(messages) != 4 {
		t.Fatalf(
			"expected 4 messages, got %d",
			len(messages),
		)
	}

	if messages[0].Role != "user" ||
		messages[0].Content != "hello" {
		t.Fatalf("unexpected first message: %+v", messages[0])
	}

	if messages[1].Role != "assistant" ||
		messages[1].Content != "response" {
		t.Fatalf("unexpected first response: %+v", messages[1])
	}

	if messages[2].Role != "user" ||
		messages[2].Content != "how are you?" {
		t.Fatalf("unexpected second message: %+v", messages[2])
	}

	if messages[3].Role != "assistant" ||
		messages[3].Content != "response" {
		t.Fatalf("unexpected second response: %+v", messages[3])
	}
}
func TestConversationEngineLLMErrorDoesNotStoreResponse(t *testing.T) {
	fakeLLM := &testLLM{
		err: errors.New("llm failed"),
	}

	llm := &LLMEngine{
		llm: fakeLLM,
	}

	engine := NewConversationEngine(llm, nil)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "error-call",
			Transcript: neurocall.Transcript{
				Text: "hello",
			},
		},
	)

	if err == nil {
		t.Fatal("expected LLM error")
	}

	conversation, err := engine.GetOrCreate("error-call")
	if err != nil {
		t.Fatal(err)
	}

	messages := conversation.Messages()

	if len(messages) != 1 {
		t.Fatalf(
			"expected 1 message, got %d",
			len(messages),
		)
	}

	if messages[0].Role != "user" {
		t.Fatalf(
			"expected user message, got %s",
			messages[0].Role,
		)
	}

	if messages[0].Content != "hello" {
		t.Fatalf(
			"expected hello, got %s",
			messages[0].Content,
		)
	}
}
func TestConversationEngineHandleTranscriptWithoutLLM(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "no-llm-call",
			Transcript: neurocall.Transcript{
				Text: "hello",
			},
		},
	)

	if err == nil {
		t.Fatal("expected LLM configuration error")
	}

	if err.Error() != "LLM is not configured" {
		t.Fatalf(
			"expected LLM is not configured, got %v",
			err,
		)
	}

	conversation, err := engine.GetOrCreate("no-llm-call")
	if err != nil {
		t.Fatal(err)
	}

	messages := conversation.Messages()

	if len(messages) != 1 {
		t.Fatalf(
			"expected user message to remain stored, got %d messages",
			len(messages),
		)
	}

	if messages[0].Role != "user" {
		t.Fatalf(
			"expected user message, got %s",
			messages[0].Role,
		)
	}
}
func TestConversationMessagesReturnsCopy(t *testing.T) {
	conversation := NewConversation()

	conversation.AddMessage(neurocall.Message{
		Role:    "user",
		Content: "hello",
	})

	messages := conversation.Messages()

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	messages[0].Content = "modified outside"

	original := conversation.Messages()

	if original[0].Content != "hello" {
		t.Fatalf(
			"conversation state was modified through returned slice: %s",
			original[0].Content,
		)
	}
}
func TestConversationConcurrentAddMessage(t *testing.T) {
	conversation := NewConversation()

	const workers = 50

	var wg sync.WaitGroup
	wg.Add(workers)

	for i := 0; i < workers; i++ {
		go func(i int) {
			defer wg.Done()

			conversation.AddMessage(neurocall.Message{
				Role:    "user",
				Content: fmt.Sprintf("message-%d", i),
			})
		}(i)
	}

	wg.Wait()

	messages := conversation.Messages()

	if len(messages) != workers {
		t.Fatalf(
			"expected %d messages, got %d",
			workers,
			len(messages),
		)
	}
}
func TestConversationAddMessagePreservesOrder(t *testing.T) {
	conversation := NewConversation()

	expected := []string{
		"first",
		"second",
		"third",
		"fourth",
	}

	for _, content := range expected {
		conversation.AddMessage(neurocall.Message{
			Role:    "user",
			Content: content,
		})
	}

	messages := conversation.Messages()

	if len(messages) != len(expected) {
		t.Fatalf(
			"expected %d messages, got %d",
			len(expected),
			len(messages),
		)
	}

	for i, want := range expected {
		if messages[i].Content != want {
			t.Fatalf(
				"message %d: expected %q, got %q",
				i,
				want,
				messages[i].Content,
			)
		}
	}
}
func TestConversationEngineConcurrentDifferentCalls(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	const workers = 50

	var wg sync.WaitGroup
	wg.Add(workers)

	for i := 0; i < workers; i++ {
		go func(i int) {
			defer wg.Done()

			callID := fmt.Sprintf("call-%d", i)

			conversation, err := engine.GetOrCreate(callID)
			if err != nil {
				t.Errorf("call %s: %v", callID, err)
				return
			}

			if conversation == nil {
				t.Errorf("call %s: expected conversation", callID)
				return
			}

			conversation.AddMessage(neurocall.Message{
				Role:    "user",
				Content: callID,
			})
		}(i)
	}

	wg.Wait()

	for i := 0; i < workers; i++ {
		callID := fmt.Sprintf("call-%d", i)

		conversation, err := engine.GetOrCreate(callID)
		if err != nil {
			t.Fatal(err)
		}

		messages := conversation.Messages()

		if len(messages) != 1 {
			t.Fatalf(
				"%s: expected 1 message, got %d",
				callID,
				len(messages),
			)
		}

		if messages[0].Content != callID {
			t.Fatalf(
				"%s: expected message %q, got %q",
				callID,
				callID,
				messages[0].Content,
			)
		}
	}
}
func TestConversationEngineConcurrentSameCall(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	const workers = 50

	results := make(chan *Conversation, workers)

	var wg sync.WaitGroup
	wg.Add(workers)

	for i := 0; i < workers; i++ {
		go func() {
			defer wg.Done()

			conversation, err := engine.GetOrCreate("same-call")
			if err != nil {
				t.Errorf("GetOrCreate failed: %v", err)
				return
			}

			results <- conversation
		}()
	}

	wg.Wait()
	close(results)

	var first *Conversation

	for conversation := range results {
		if conversation == nil {
			t.Fatal("expected conversation, got nil")
		}

		if first == nil {
			first = conversation
			continue
		}

		if conversation != first {
			t.Fatal(
				"GetOrCreate returned different Conversation instances for the same CallID",
			)
		}
	}
}
func TestConversationEngineHandleTranscriptEmptyCallID2(t *testing.T) {
	fakeLLM := &testLLM{}

	llm := &LLMEngine{
		llm: fakeLLM,
	}

	engine := NewConversationEngine(llm, nil)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "",
			Transcript: neurocall.Transcript{
				Text: "hello",
			},
		},
	)

	if err == nil {
		t.Fatal("expected error for empty call ID")
	}

	if err.Error() != "call ID is empty" {
		t.Fatalf(
			"expected call ID is empty, got %v",
			err,
		)
	}
}
func TestConversationEngineHandleTranscriptWhitespace(t *testing.T) {
	fakeLLM := &testLLM{
		response: neurocall.Message{
			Role:    "assistant",
			Content: "should not be called",
		},
	}

	llm := &LLMEngine{
		llm: fakeLLM,
	}

	engine := NewConversationEngine(llm, nil)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "whitespace-call",
			Transcript: neurocall.Transcript{
				Text: "   \t\n   ",
			},
		},
	)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	conversation, err := engine.GetOrCreate("whitespace-call")
	if err != nil {
		t.Fatal(err)
	}

	messages := conversation.Messages()

	if len(messages) != 0 {
		t.Fatalf(
			"expected no messages, got %d",
			len(messages),
		)
	}
}
func TestConversationEngineSeparateCalls(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	first, err := engine.GetOrCreate("call-1")
	if err != nil {
		t.Fatal(err)
	}

	second, err := engine.GetOrCreate("call-2")
	if err != nil {
		t.Fatal(err)
	}

	if first == nil || second == nil {
		t.Fatal("expected both conversations to be non-nil")
	}

	if first == second {
		t.Fatal("different CallIDs must have different conversations")
	}

	first.AddMessage(neurocall.Message{
		Role:    "user",
		Content: "hello from call 1",
	})

	if len(first.Messages()) != 1 {
		t.Fatalf(
			"expected call-1 to have 1 message, got %d",
			len(first.Messages()),
		)
	}

	if len(second.Messages()) != 0 {
		t.Fatalf(
			"expected call-2 to have 0 messages, got %d",
			len(second.Messages()),
		)
	}
}
func TestTTSEngineSetTTSNil(t *testing.T) {
	engine := NewTTSEngine(&testTTS{})

	engine.SetTTS(nil)

	_, err := engine.Synthesize(
		context.Background(),
		"hello",
	)

	if err == nil {
		t.Fatal("expected error when TTS is nil")
	}

	if err.Error() != "TTS is not configured" {
		t.Fatalf(
			"expected TTS is not configured, got %v",
			err,
		)
	}
}
func TestTTSEngineSynthesizeNilContext(t *testing.T) {
	engine := NewTTSEngine(&testTTS{
		audio: neurocall.AudioStreamData{
			Data: []byte{1, 2, 3, 4},
			Format: neurocall.AudioFormat{
				SampleRate: 8000,
			},
		},
	})

	audio, err := engine.Synthesize(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}

	if len(audio.Data) == 0 {
		t.Fatal("expected audio data")
	}
}
func TestVoiceResponseEngineHandleResponseEmptyCallID(t *testing.T) {
	tts := NewTTSEngine(&testTTS{})

	engine := NewVoiceResponseEngine(tts, nil)

	err := engine.HandleResponse(
		LLMResponseEvent{
			CallID: "",
			Message: neurocall.Message{
				Role:    "assistant",
				Content: "hello",
			},
		},
	)

	if err == nil {
		t.Fatal("expected error for empty call ID")
	}

	if err.Error() != "call ID is empty" {
		t.Fatalf(
			"expected call ID is empty, got %v",
			err,
		)
	}
}
func TestVoiceResponseEngineHandleResponseEmptyText(t *testing.T) {
	engine := NewVoiceResponseEngine(
		NewTTSEngine(&testTTS{}),
		nil,
	)

	err := engine.HandleResponse(
		LLMResponseEvent{
			CallID: "call-empty-text",
			Message: neurocall.Message{
				Role:    "assistant",
				Content: "   \t\n ",
			},
		},
	)

	if err != nil {
		t.Fatalf(
			"expected no error, got %v",
			err,
		)
	}
}
func TestVoiceResponseEngineHandleResponseCallNotFound(t *testing.T) {
	engine := NewVoiceResponseEngine(
		NewTTSEngine(&testTTS{}),
		nil,
	)

	err := engine.HandleResponse(
		LLMResponseEvent{
			CallID: "missing-call",
			Message: neurocall.Message{
				Role:    "assistant",
				Content: "hello",
			},
		},
	)

	if !errors.Is(err, neurocall.ErrCallNotFound) {
		t.Fatalf(
			"expected ErrCallNotFound, got %v",
			err,
		)
	}
}
func TestVoiceResponseEngineHandleResponseClosedCall(t *testing.T) {
	engine := NewVoiceResponseEngine(
		NewTTSEngine(&testTTS{}),
		nil,
	)

	call := &SIPCall{
		CallID: "closed-call",
	}

	engine.RegisterCall(call)

	if err := call.CloseResources(); err != nil {
		t.Fatal(err)
	}

	err := engine.HandleResponse(
		LLMResponseEvent{
			CallID: "closed-call",
			Message: neurocall.Message{
				Role:    "assistant",
				Content: "hello",
			},
		},
	)

	if !errors.Is(err, neurocall.ErrCallClosed) {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestVoiceResponseEngineRegisterNilCall(t *testing.T) {
	engine := NewVoiceResponseEngine(
		NewTTSEngine(&testTTS{}),
		nil,
	)

	engine.RegisterCall(nil)

	if len(engine.calls) != 0 {
		t.Fatalf(
			"expected no calls to be registered, got %d",
			len(engine.calls),
		)
	}
}
func TestVoiceResponseEngineRemoveCall(t *testing.T) {
	engine := NewVoiceResponseEngine(
		NewTTSEngine(&testTTS{}),
		nil,
	)

	call := &SIPCall{
		CallID: "remove-call",
	}

	engine.RegisterCall(call)

	engine.RemoveCall("remove-call")

	if _, ok := engine.calls["remove-call"]; ok {
		t.Fatal("call should have been removed")
	}

	err := engine.HandleResponse(
		LLMResponseEvent{
			CallID: "remove-call",
			Message: neurocall.Message{
				Role:    "assistant",
				Content: "hello",
			},
		},
	)

	if !errors.Is(err, neurocall.ErrCallNotFound) {
		t.Fatalf(
			"expected ErrCallNotFound, got %v",
			err,
		)
	}
}
func TestVoiceResponseEngineRegisterCallOverwrite(t *testing.T) {
	engine := NewVoiceResponseEngine(
		NewTTSEngine(&testTTS{}),
		nil,
	)

	first := &SIPCall{
		CallID: "same-call",
	}

	second := &SIPCall{
		CallID: "same-call",
	}

	engine.RegisterCall(first)
	engine.RegisterCall(second)

	if len(engine.calls) != 1 {
		t.Fatalf(
			"expected 1 registered call, got %d",
			len(engine.calls),
		)
	}

	call, err := engine.GetCall("same-call")
	if err != nil {
		t.Fatal(err)
	}

	if call != second {
		t.Fatal("expected second call to replace the first")
	}
}
func TestVoiceResponseEngineRemoveMissingCall(t *testing.T) {
	engine := NewVoiceResponseEngine(
		NewTTSEngine(&testTTS{}),
		nil,
	)

	err := engine.RemoveCall("missing-call")

	if !errors.Is(err, neurocall.ErrCallNotFound) {
		t.Fatalf(
			"expected ErrCallNotFound, got %v",
			err,
		)
	}
}
func TestVoiceResponseEngineGetMissingCall(t *testing.T) {
	engine := NewVoiceResponseEngine(
		NewTTSEngine(&testTTS{}),
		nil,
	)

	call, err := engine.GetCall("missing-call")

	if !errors.Is(err, neurocall.ErrCallNotFound) {
		t.Fatalf(
			"expected ErrCallNotFound, got %v",
			err,
		)
	}

	if call != nil {
		t.Fatal("expected nil call")
	}
}
func TestVoiceResponseEngineGetCall(t *testing.T) {
	engine := NewVoiceResponseEngine(
		NewTTSEngine(&testTTS{}),
		nil,
	)

	call := &SIPCall{
		CallID: "get-call",
	}

	engine.RegisterCall(call)

	got, err := engine.GetCall("get-call")
	if err != nil {
		t.Fatal(err)
	}

	if got != call {
		t.Fatal("expected GetCall to return the registered call")
	}
}
func TestConversationEngineGetOrCreateConcurrent(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	const workers = 20

	results := make(chan *Conversation, workers)
	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			conversation, err := engine.GetOrCreate("concurrent-call")
			if err != nil {
				t.Errorf("GetOrCreate failed: %v", err)
				return
			}

			results <- conversation
		}()
	}

	wg.Wait()
	close(results)

	var first *Conversation

	for conversation := range results {
		if first == nil {
			first = conversation
			continue
		}

		if conversation != first {
			t.Fatal("expected all goroutines to receive the same Conversation")
		}
	}

	if first == nil {
		t.Fatal("expected a conversation")
	}
}
func TestConversationConcurrentMessages(t *testing.T) {
	conversation := &Conversation{}

	const workers = 50

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)

		go func(i int) {
			defer wg.Done()

			conversation.AddMessage(neurocall.Message{
				Role:    "user",
				Content: fmt.Sprintf("message-%d", i),
			})
		}(i)
	}

	wg.Wait()

	messages := conversation.Messages()

	if len(messages) != workers {
		t.Fatalf(
			"expected %d messages, got %d",
			workers,
			len(messages),
		)
	}
}
func TestConversationEngineGetOrCreateMultipleCallsConcurrent(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	const workers = 50

	var wg sync.WaitGroup
	results := make(chan *Conversation, workers)

	for i := 0; i < workers; i++ {
		wg.Add(1)

		go func(i int) {
			defer wg.Done()

			callID := fmt.Sprintf("call-%d", i)

			conversation, err := engine.GetOrCreate(callID)
			if err != nil {
				t.Errorf(
					"GetOrCreate(%s) failed: %v",
					callID,
					err,
				)
				return
			}

			results <- conversation
		}(i)
	}

	wg.Wait()
	close(results)

	if len(results) != workers {
		t.Fatalf(
			"expected %d conversations, got %d",
			workers,
			len(results),
		)
	}

	if len(engine.conversations) != workers {
		t.Fatalf(
			"expected %d stored conversations, got %d",
			workers,
			len(engine.conversations),
		)
	}

	seen := make(map[*Conversation]bool)

	for conversation := range results {
		if conversation == nil {
			t.Fatal("conversation should not be nil")
		}

		if seen[conversation] {
			t.Fatal("same Conversation returned for different CallIDs")
		}

		seen[conversation] = true
	}
}
func TestConversationEngineGetOrCreateEmptyCallID(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	conversation, err := engine.GetOrCreate("")

	if err == nil {
		t.Fatal("expected error for empty call ID")
	}

	if conversation != nil {
		t.Fatal("expected nil conversation")
	}

	if len(engine.conversations) != 0 {
		t.Fatalf(
			"expected no conversations, got %d",
			len(engine.conversations),
		)
	}
}
func TestConversationEngineGetOrCreateConcurrent2(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	const workers = 20

	results := make(chan *Conversation, workers)
	errs := make(chan error, workers)

	for i := 0; i < workers; i++ {
		go func() {
			conversation, err := engine.GetOrCreate("concurrent-call")

			if err != nil {
				errs <- err
				return
			}

			results <- conversation
		}()
	}

	var first *Conversation

	for i := 0; i < workers; i++ {
		select {
		case err := <-errs:
			t.Fatal(err)

		case conversation := <-results:
			if conversation == nil {
				t.Fatal("expected non-nil conversation")
			}

			if first == nil {
				first = conversation
				continue
			}

			if conversation != first {
				t.Fatal("expected all goroutines to receive the same conversation")
			}
		}
	}
}
func TestConversationEngineHandleTranscriptNewCall(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "new-transcript-call",
			Transcript: neurocall.Transcript{
				Text: "hello from caller",
			},
		},
	)

	if err == nil {
		t.Fatal("expected LLM configuration error")
	}

	if err.Error() != "LLM is not configured" {
		t.Fatalf(
			"expected LLM is not configured, got %v",
			err,
		)
	}

	conversation, err := engine.GetOrCreate("new-transcript-call")
	if err != nil {
		t.Fatal(err)
	}

	messages := conversation.Messages()

	if len(messages) != 1 {
		t.Fatalf(
			"expected 1 message, got %d",
			len(messages),
		)
	}

	if messages[0].Role != "user" {
		t.Fatalf(
			"expected role user, got %q",
			messages[0].Role,
		)
	}

	if messages[0].Content != "hello from caller" {
		t.Fatalf(
			"expected transcript %q, got %q",
			"hello from caller",
			messages[0].Content,
		)
	}
}
func TestConversationEngineHandleTranscriptEmptyText(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "empty-transcript-call",
			Transcript: neurocall.Transcript{
				Text: "   \t\n ",
			},
		},
	)

	if err != nil {
		t.Fatalf(
			"expected no error, got %v",
			err,
		)
	}

	if len(engine.conversations) != 0 {
		t.Fatalf(
			"expected no conversation to be created, got %d",
			len(engine.conversations),
		)
	}
}
func TestConversationEngineHandleTranscriptEmptyCallID(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: "",
			Transcript: neurocall.Transcript{
				Text: "hello",
			},
		},
	)

	if err == nil {
		t.Fatal("expected error for empty call ID")
	}

	if len(engine.conversations) != 0 {
		t.Fatalf(
			"expected no conversation to be created, got %d",
			len(engine.conversations),
		)
	}
}
func TestConversationEngineHandleTranscriptMultiple(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	callID := "multi-transcript-call"

	err := engine.HandleTranscript(
		TranscriptEvent{
			CallID: callID,
			Transcript: neurocall.Transcript{
				Text: "hello",
			},
		},
	)

	if err == nil {
		t.Fatal("expected LLM configuration error")
	}

	err = engine.HandleTranscript(
		TranscriptEvent{
			CallID: callID,
			Transcript: neurocall.Transcript{
				Text: "how are you",
			},
		},
	)

	if err == nil {
		t.Fatal("expected LLM configuration error")
	}

	if len(engine.conversations) != 1 {
		t.Fatalf(
			"expected 1 conversation, got %d",
			len(engine.conversations),
		)
	}

	conversation, err := engine.GetOrCreate(callID)
	if err != nil {
		t.Fatal(err)
	}

	messages := conversation.Messages()

	if len(messages) != 2 {
		t.Fatalf(
			"expected 2 messages, got %d",
			len(messages),
		)
	}

	if messages[0].Content != "hello" {
		t.Fatalf(
			"expected first message %q, got %q",
			"hello",
			messages[0].Content,
		)
	}

	if messages[1].Content != "how are you" {
		t.Fatalf(
			"expected second message %q, got %q",
			"how are you",
			messages[1].Content,
		)
	}
}
func TestConversationMessagesOrderAndRole(t *testing.T) {
	conversation := &Conversation{}

	conversation.AddMessage(neurocall.Message{
		Role:    "user",
		Content: "hello",
	})

	conversation.AddMessage(neurocall.Message{
		Role:    "assistant",
		Content: "hi, how can I help?",
	})

	conversation.AddMessage(neurocall.Message{
		Role:    "user",
		Content: "what time is it?",
	})

	messages := conversation.Messages()

	if len(messages) != 3 {
		t.Fatalf(
			"expected 3 messages, got %d",
			len(messages),
		)
	}

	expected := []neurocall.Message{
		{
			Role:    "user",
			Content: "hello",
		},
		{
			Role:    "assistant",
			Content: "hi, how can I help?",
		},
		{
			Role:    "user",
			Content: "what time is it?",
		},
	}

	for i := range expected {
		if messages[i].Role != expected[i].Role {
			t.Fatalf(
				"message %d: expected role %q, got %q",
				i,
				expected[i].Role,
				messages[i].Role,
			)
		}

		if messages[i].Content != expected[i].Content {
			t.Fatalf(
				"message %d: expected content %q, got %q",
				i,
				expected[i].Content,
				messages[i].Content,
			)
		}
	}
}
func TestConversationMessagesIsolation(t *testing.T) {
	conversation := &Conversation{}

	conversation.AddMessage(neurocall.Message{
		Role:    "user",
		Content: "original",
	})

	messages := conversation.Messages()

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	messages[0].Content = "modified"

	original := conversation.Messages()

	if original[0].Content != "original" {
		t.Fatalf(
			"conversation was modified through returned Messages slice: got %q",
			original[0].Content,
		)
	}
}
func TestCallMemorySetGet(t *testing.T) {
	memory := &CallMemory{
		values: make(map[string]any),
	}

	memory.Set("name", "Daniyal")

	value, ok := memory.Get("name")

	if !ok {
		t.Fatal("expected value to exist")
	}

	if value != "Daniyal" {
		t.Fatalf(
			"expected %q, got %v",
			"Daniyal",
			value,
		)
	}
}
func TestCallMemoryGetMissing(t *testing.T) {
	memory := &CallMemory{
		values: make(map[string]any),
	}

	value, ok := memory.Get("missing")

	if ok {
		t.Fatal("expected ok to be false for missing key")
	}

	if value != nil {
		t.Fatalf(
			"expected nil value, got %v",
			value,
		)
	}
}
func TestCallMemoryDelete(t *testing.T) {
	memory := &CallMemory{
		values: make(map[string]any),
	}

	memory.Set("name", "Daniyal")

	memory.Delete("name")

	value, ok := memory.Get("name")

	if ok {
		t.Fatal("expected key to be deleted")
	}

	if value != nil {
		t.Fatalf(
			"expected nil value after delete, got %v",
			value,
		)
	}
}
func TestCallMemoryClear(t *testing.T) {
	memory := &CallMemory{
		values: make(map[string]any),
	}

	memory.Set("name", "Daniyal")
	memory.Set("age", 30)
	memory.Set("active", true)

	memory.Clear()

	if len(memory.values) != 0 {
		t.Fatalf(
			"expected memory to be empty, got %d values",
			len(memory.values),
		)
	}

	for _, key := range []string{"name", "age", "active"} {
		value, ok := memory.Get(key)

		if ok {
			t.Fatalf("expected %q to be deleted", key)
		}

		if value != nil {
			t.Fatalf(
				"expected nil value for %q, got %v",
				key,
				value,
			)
		}
	}
}
func TestCallMemoryConcurrentAccess(t *testing.T) {
	memory := &CallMemory{
		values: make(map[string]any),
	}

	const workers = 50

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)

		go func(i int) {
			defer wg.Done()

			key := fmt.Sprintf("key-%d", i)

			memory.Set(key, i)

			value, ok := memory.Get(key)
			if !ok {
				t.Errorf("expected key %q to exist", key)
				return
			}

			if value != i {
				t.Errorf(
					"key %q: expected %d, got %v",
					key,
					i,
					value,
				)
			}
		}(i)
	}

	wg.Wait()
}
func TestCallMemoryConcurrentMutation(t *testing.T) {
	memory := &CallMemory{
		values: make(map[string]any),
	}

	const workers = 30

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)

		go func(i int) {
			defer wg.Done()

			key := fmt.Sprintf("key-%d", i)

			memory.Set(key, i)
			memory.Get(key)
			memory.Delete(key)
		}(i)
	}

	wg.Add(1)

	go func() {
		defer wg.Done()

		for i := 0; i < workers; i++ {
			memory.Set(
				fmt.Sprintf("clear-key-%d", i),
				i,
			)
		}

		memory.Clear()
	}()

	wg.Wait()
}
func TestCallMemoryDifferentValueTypes(t *testing.T) {
	memory := &CallMemory{
		values: make(map[string]any),
	}

	memory.Set("string", "hello")
	memory.Set("int", 42)
	memory.Set("bool", true)
	memory.Set("float", 3.14)
	memory.Set("slice", []string{"a", "b"})
	memory.Set("nil", nil)

	if got, ok := memory.Get("string"); !ok || got != "hello" {
		t.Fatalf("unexpected string value: %v", got)
	}

	if got, ok := memory.Get("int"); !ok || got != 42 {
		t.Fatalf("unexpected int value: %v", got)
	}

	if got, ok := memory.Get("bool"); !ok || got != true {
		t.Fatalf("unexpected bool value: %v", got)
	}

	if got, ok := memory.Get("float"); !ok || got != 3.14 {
		t.Fatalf("unexpected float value: %v", got)
	}

	got, ok := memory.Get("slice")
	if !ok {
		t.Fatal("expected slice value to exist")
	}

	slice, ok := got.([]string)
	if !ok {
		t.Fatalf("expected []string, got %T", got)
	}

	if len(slice) != 2 || slice[0] != "a" || slice[1] != "b" {
		t.Fatalf("unexpected slice value: %v", slice)
	}

	got, ok = memory.Get("nil")
	if !ok {
		t.Fatal("expected nil value to exist")
	}

	if got != nil {
		t.Fatalf("expected nil, got %v", got)
	}
}
func TestCallMemoryDeleteMissing(t *testing.T) {
	memory := &CallMemory{
		values: make(map[string]any),
	}

	memory.Set("existing", "value")

	memory.Delete("missing")

	value, ok := memory.Get("existing")

	if !ok {
		t.Fatal("existing key should remain after deleting missing key")
	}

	if value != "value" {
		t.Fatalf(
			"expected existing value %q, got %v",
			"value",
			value,
		)
	}

	if len(memory.values) != 1 {
		t.Fatalf(
			"expected 1 value, got %d",
			len(memory.values),
		)
	}
}
func TestCallMemoryClearEmptyAndRepeated(t *testing.T) {
	memory := &CallMemory{
		values: make(map[string]any),
	}

	memory.Clear()

	if len(memory.values) != 0 {
		t.Fatalf("expected empty memory, got %d values", len(memory.values))
	}

	memory.Set("key", "value")

	memory.Clear()
	memory.Clear()
	memory.Clear()

	if len(memory.values) != 0 {
		t.Fatalf(
			"expected memory to remain empty, got %d values",
			len(memory.values),
		)
	}

	value, ok := memory.Get("key")

	if ok {
		t.Fatal("expected key to remain deleted")
	}

	if value != nil {
		t.Fatalf("expected nil value, got %v", value)
	}
}
func TestConversationMemoryIsolation(t *testing.T) {
	first := &Conversation{
		memory: &CallMemory{
			values: make(map[string]any),
		},
	}

	second := &Conversation{
		memory: &CallMemory{
			values: make(map[string]any),
		},
	}

	first.memory.Set("name", "Daniyal")
	second.memory.Set("name", "Other")

	firstValue, ok := first.memory.Get("name")
	if !ok {
		t.Fatal("expected first conversation memory to contain name")
	}

	secondValue, ok := second.memory.Get("name")
	if !ok {
		t.Fatal("expected second conversation memory to contain name")
	}

	if firstValue != "Daniyal" {
		t.Fatalf(
			"expected first memory value %q, got %v",
			"Daniyal",
			firstValue,
		)
	}

	if secondValue != "Other" {
		t.Fatalf(
			"expected second memory value %q, got %v",
			"Other",
			secondValue,
		)
	}

	first.memory.Delete("name")

	if _, ok := first.memory.Get("name"); ok {
		t.Fatal("expected first memory key to be deleted")
	}

	if value, ok := second.memory.Get("name"); !ok || value != "Other" {
		t.Fatal("second conversation memory should remain unchanged")
	}
}
func TestConversationEngineGetOrCreateInitializesMemory(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	conversation, err := engine.GetOrCreate("memory-call")
	if err != nil {
		t.Fatal(err)
	}

	if conversation == nil {
		t.Fatal("expected conversation")
	}

	if conversation.memory == nil {
		t.Fatal("expected Conversation memory to be initialized")
	}

	if conversation.memory.values == nil {
		t.Fatal("expected CallMemory values map to be initialized")
	}

	conversation.memory.Set("test", "value")

	value, ok := conversation.memory.Get("test")
	if !ok {
		t.Fatal("expected memory value to exist")
	}

	if value != "value" {
		t.Fatalf(
			"expected %q, got %v",
			"value",
			value,
		)
	}
}
func TestConversationEngineGetOrCreatePreservesMemory(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	first, err := engine.GetOrCreate("persistent-memory-call")
	if err != nil {
		t.Fatal(err)
	}

	first.memory.Set("name", "Daniyal")

	second, err := engine.GetOrCreate("persistent-memory-call")
	if err != nil {
		t.Fatal(err)
	}

	if second != first {
		t.Fatal("expected the same Conversation instance")
	}

	value, ok := second.memory.Get("name")
	if !ok {
		t.Fatal("expected memory value to persist")
	}

	if value != "Daniyal" {
		t.Fatalf(
			"expected %q, got %v",
			"Daniyal",
			value,
		)
	}
}
func TestConversationMemoryClearKeepsConversation(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	conversation, err := engine.GetOrCreate("clear-memory-call")
	if err != nil {
		t.Fatal(err)
	}

	conversation.memory.Set("name", "Daniyal")
	conversation.memory.Set("language", "fa")

	conversation.memory.Clear()

	if _, ok := conversation.memory.Get("name"); ok {
		t.Fatal("expected name to be cleared")
	}

	if _, ok := conversation.memory.Get("language"); ok {
		t.Fatal("expected language to be cleared")
	}

	same, err := engine.GetOrCreate("clear-memory-call")
	if err != nil {
		t.Fatal(err)
	}

	if same != conversation {
		t.Fatal("expected Conversation to remain after memory Clear")
	}

	if same.memory == nil {
		t.Fatal("expected Conversation memory to remain initialized")
	}
}
func TestConversationEngineConcurrentMemoryAccess(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	const workers = 50
	callID := "concurrent-memory-call"

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)

		go func(i int) {
			defer wg.Done()

			conversation, err := engine.GetOrCreate(callID)
			if err != nil {
				t.Errorf("GetOrCreate failed: %v", err)
				return
			}

			key := fmt.Sprintf("key-%d", i)

			conversation.memory.Set(key, i)

			value, ok := conversation.memory.Get(key)
			if !ok {
				t.Errorf("expected key %q to exist", key)
				return
			}

			if value != i {
				t.Errorf(
					"key %q: expected %d, got %v",
					key,
					i,
					value,
				)
			}
		}(i)
	}

	wg.Wait()

	conversation, err := engine.GetOrCreate(callID)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < workers; i++ {
		key := fmt.Sprintf("key-%d", i)

		value, ok := conversation.memory.Get(key)
		if !ok {
			t.Fatalf("expected key %q to exist", key)
		}

		if value != i {
			t.Fatalf(
				"key %q: expected %d, got %v",
				key,
				i,
				value,
			)
		}
	}
}
func TestConversationMemoryClearPreservesMessages(t *testing.T) {
	engine := NewConversationEngine(nil, nil)

	conversation, err := engine.GetOrCreate("memory-message-call")
	if err != nil {
		t.Fatal(err)
	}

	conversation.memory.Set("name", "Daniyal")

	conversation.AddMessage(neurocall.Message{
		Role:    "user",
		Content: "hello",
	})

	conversation.memory.Clear()

	messages := conversation.Messages()

	if len(messages) != 1 {
		t.Fatalf(
			"expected 1 message after memory clear, got %d",
			len(messages),
		)
	}

	if messages[0].Content != "hello" {
		t.Fatalf(
			"expected message %q, got %q",
			"hello",
			messages[0].Content,
		)
	}

	conversation.AddMessage(neurocall.Message{
		Role:    "user",
		Content: "second message",
	})

	messages = conversation.Messages()

	if len(messages) != 2 {
		t.Fatalf(
			"expected 2 messages, got %d",
			len(messages),
		)
	}

	if messages[1].Content != "second message" {
		t.Fatalf(
			"expected second message %q, got %q",
			"second message",
			messages[1].Content,
		)
	}
}
func TestCallMemorySetAfterClear(t *testing.T) {
	memory := &CallMemory{
		values: make(map[string]any),
	}

	memory.Set("old", "value")
	memory.Clear()

	if _, ok := memory.Get("old"); ok {
		t.Fatal("expected old value to be cleared")
	}

	memory.Set("new", "new-value")

	value, ok := memory.Get("new")
	if !ok {
		t.Fatal("expected new value to exist after Clear")
	}

	if value != "new-value" {
		t.Fatalf(
			"expected %q, got %v",
			"new-value",
			value,
		)
	}
}
func TestSIPCallCloseResourcesIdempotent(t *testing.T) {
	call := &SIPCall{}

	call.CloseResources()
	call.CloseResources()

}
func TestSIPCallCloseResourcesClearsReferences(t *testing.T) {
	call := &SIPCall{
		CallID: "close-resources-test",
	}

	call.CloseResources()

	if call.Pipeline != nil {
		t.Fatal("expected Pipeline to be nil")
	}

	if call.RTP != nil {
		t.Fatal("expected RTP to be nil")
	}

	if call.TTS != nil {
		t.Fatal("expected TTS to be nil")
	}

	if call.STT != nil {
		t.Fatal("expected STT to be nil")
	}

	if call.Conversation != nil {
		t.Fatal("expected Conversation to be nil")
	}
}
func TestSIPCallCloseResourcesClosesPipeline(t *testing.T) {
	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})

	call := &SIPCall{
		CallID:   "close-pipeline-test",
		Pipeline: pipeline,
	}

	call.CloseResources()

	if call.Pipeline != nil {
		t.Fatal("expected Pipeline to be cleared")
	}
}
func TestSIPCallCloseResourcesPipelineIdempotent(t *testing.T) {
	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})

	call := &SIPCall{
		CallID:   "pipeline-idempotent-test",
		Pipeline: pipeline,
	}

	call.CloseResources()
	call.CloseResources()

	if call.Pipeline != nil {
		t.Fatal("expected Pipeline to remain nil")
	}
}
func TestSIPCallCloseResourcesClosesRTP(t *testing.T) {
	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		neurocall.Codec{
			Name:        "PCMU",
			PayloadType: 0,
			ClockRate:   8000,
			Channels:    1,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	call := &SIPCall{
		CallID: "close-rtp-test",
		RTP:    rtp,
	}

	conn := rtp.conn

	call.CloseResources()

	if call.RTP != nil {
		t.Fatal("expected RTP to be cleared")
	}

	rtp.mu.RLock()
	closed := rtp.closed
	rtp.mu.RUnlock()

	if !closed {
		t.Fatal("expected RTP session to be marked closed")
	}

	if conn == nil {
		t.Fatal("expected RTP connection to have existed")
	}

	_, err = conn.WriteToUDP(
		[]byte{0},
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: 9000,
		},
	)

	if err == nil {
		t.Fatal("expected write on closed RTP connection to fail")
	}
}
func TestSIPCallCloseResourcesRTPAndPipeline(t *testing.T) {
	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		neurocall.Codec{
			Name:        "PCMU",
			PayloadType: 0,
			ClockRate:   8000,
			Channels:    1,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})

	call := &SIPCall{
		CallID:   "close-all-media-test",
		RTP:      rtp,
		Pipeline: pipeline,
	}

	call.CloseResources()

	if call.RTP != nil {
		t.Fatal("expected RTP to be cleared")
	}

	if call.Pipeline != nil {
		t.Fatal("expected Pipeline to be cleared")
	}

	rtp.mu.RLock()
	rtpClosed := rtp.closed
	rtp.mu.RUnlock()

	if !rtpClosed {
		t.Fatal("expected RTP session to be closed")
	}
}
func TestSIPCallCloseResourcesTTSRTPAndPipeline(t *testing.T) {
	rtp, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		neurocall.Codec{
			Name:        "PCMU",
			PayloadType: 0,
			ClockRate:   8000,
			Channels:    1,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})

	ttsEngine := NewTTSEngine(nil)

	ttsPlayback, err := NewTTSPlayback(ttsEngine)
	if err != nil {
		t.Fatal(err)
	}

	call := &SIPCall{
		CallID:   "close-media-test",
		TTS:      ttsPlayback,
		RTP:      rtp,
		Pipeline: pipeline,
	}

	call.CloseResources()

	if call.TTS != nil {
		t.Fatal("expected TTS to be cleared")
	}

	if call.RTP != nil {
		t.Fatal("expected RTP to be cleared")
	}

	if call.Pipeline != nil {
		t.Fatal("expected Pipeline to be cleared")
	}

	rtp.mu.RLock()
	rtpClosed := rtp.closed
	rtp.mu.RUnlock()

	if !rtpClosed {
		t.Fatal("expected RTP session to be closed")
	}
}
func TestCallManagerAddGetRemove(t *testing.T) {
	manager := NewCallManager(10000, 10100)

	call := &SIPCall{
		CallID: "manager-lifecycle-call",
	}

	if err := manager.Add(call); err != nil {
		t.Fatal(err)
	}

	got, err := manager.Get(call.CallID)
	if err != nil {
		t.Fatal(err)
	}

	if got != call {
		t.Fatal("expected Get to return the same SIPCall")
	}

	calls := manager.List()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}

	if err := manager.Remove(call.CallID); err != nil {
		t.Fatal(err)
	}

	if _, err := manager.Get(call.CallID); err == nil {
		t.Fatal("expected Get to fail after Remove")
	}

	if len(manager.List()) != 0 {
		t.Fatal("expected CallManager to be empty after Remove")
	}
}
func TestCallManagerSessionLifecycle(t *testing.T) {
	manager := NewCallManager(10000, 10100)

	call := &SIPCall{
		CallID: "session-lifecycle-call",
	}

	if err := manager.Add(call); err != nil {
		t.Fatal(err)
	}

	session, err := manager.CreateSession(call)
	if err != nil {
		t.Fatal(err)
	}

	if session == nil {
		t.Fatal("expected session to be created")
	}

	got, err := manager.GetSession(call.CallID)
	if err != nil {
		t.Fatal(err)
	}

	if got != session {
		t.Fatal("expected GetSession to return the same session")
	}

	if err := manager.RemoveSession(call.CallID); err != nil {
		t.Fatal(err)
	}

	if _, err := manager.GetSession(call.CallID); err == nil {
		t.Fatal("expected GetSession to fail after RemoveSession")
	}
}
func TestCallManagerStartSession(t *testing.T) {
	manager := NewCallManager(10000, 10100)

	call := &SIPCall{
		CallID: "start-session-call",
	}

	if err := manager.Add(call); err != nil {
		t.Fatal(err)
	}

	session, err := manager.StartSession(call)
	if err != nil {
		t.Fatal(err)
	}

	if session == nil {
		t.Fatal("expected session")
	}

	if session.ID != call.CallID {
		t.Fatalf(
			"expected session ID %q, got %q",
			call.CallID,
			session.ID,
		)
	}

	state := session.State()

	if !state.Running {
		t.Fatal("expected session to be running")
	}

	if state.Closed {
		t.Fatal("expected session to be open")
	}

	got, err := manager.GetSession(call.CallID)
	if err != nil {
		t.Fatal(err)
	}

	if got != session {
		t.Fatal("expected GetSession to return started session")
	}
}
func TestCallManagerStartSessionWithAudio(t *testing.T) {
	manager := NewCallManager(10000, 10100)

	pipeline := NewAudioPipeline(AudioConfig{
		BufferSize: 10,
	})

	call := &SIPCall{
		CallID:   "start-session-audio-call",
		Pipeline: pipeline,
	}

	if err := manager.Add(call); err != nil {
		t.Fatal(err)
	}

	session, err := manager.StartSession(call)
	if err != nil {
		t.Fatal(err)
	}

	if session == nil {
		t.Fatal("expected session")
	}

	state := session.State()

	if !state.Running {
		t.Fatal("expected session to be running")
	}

	if state.Closed {
		t.Fatal("expected session to be open")
	}

	if session.Audio != nil {
		t.Fatal("expected session Audio to remain nil")
	}

	if err := manager.RemoveSession(call.CallID); err != nil {
		t.Fatal(err)
	}

	state = session.State()

	if !state.Closed {
		t.Fatal("expected session to be closed")
	}
}
func TestCallManagerStartSession2(t *testing.T) {
	manager := NewCallManager(10000, 10100)

	call := &SIPCall{
		CallID: "start-session-call",
	}

	if err := manager.Add(call); err != nil {
		t.Fatal(err)
	}

	session, err := manager.StartSession(call)
	if err != nil {
		t.Fatal(err)
	}

	if session == nil {
		t.Fatal("expected session")
	}

	if session.ID != call.CallID {
		t.Fatalf(
			"expected session ID %q, got %q",
			call.CallID,
			session.ID,
		)
	}

	state := session.State()

	if !state.Running {
		t.Fatal("expected session to be running")
	}

	if state.Closed {
		t.Fatal("expected session to be open")
	}

	got, err := manager.GetSession(call.CallID)
	if err != nil {
		t.Fatal(err)
	}

	if got != session {
		t.Fatal("expected GetSession to return same session")
	}

	if err := manager.RemoveSession(call.CallID); err != nil {
		t.Fatal(err)
	}

	state = session.State()

	if !state.Closed {
		t.Fatal("expected session to be closed after RemoveSession")
	}
}
func TestCallSessionConfigureConversation(t *testing.T) {
	call := &SIPCall{
		CallID: "conversation-test-call",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	llm := NewLLMEngine(nil)

	if err := session.ConfigureConversation(llm); err != nil {
		t.Fatal(err)
	}

	if session.LLM != llm {
		t.Fatal("expected LLM engine to be assigned")
	}

	if session.ConversationEngine == nil {
		t.Fatal("expected ConversationEngine to be created")
	}

	state := session.State()

	if !state.HasLLM {
		t.Fatal("expected session to report HasLLM=true")
	}
}
func TestNewCallSessionNilCall(t *testing.T) {
	session, err := NewCallSession(nil, NewEventBus())

	if err == nil {
		t.Fatal("expected error for nil call")
	}

	if session != nil {
		t.Fatal("expected nil session")
	}
}
func TestNewCallSessionEmptyCallID(t *testing.T) {
	call := &SIPCall{}

	session, err := NewCallSession(call, NewEventBus())

	if err == nil {
		t.Fatal("expected error for empty CallID")
	}

	if session != nil {
		t.Fatal("expected nil session")
	}
}
func TestNewCallSessionInitialState(t *testing.T) {
	call := &SIPCall{
		CallID: "new-session-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if session == nil {
		t.Fatal("expected session")
	}

	if session.ID != call.CallID {
		t.Fatalf(
			"expected session ID %q, got %q",
			call.CallID,
			session.ID,
		)
	}

	if session.Call != call {
		t.Fatal("expected session to reference original SIPCall")
	}

	if session.Events == nil {
		t.Fatal("expected EventBus to be initialized")
	}

	if session.Conversation == nil {
		t.Fatal("expected Conversation to be initialized")
	}

	if session.Memory == nil {
		t.Fatal("expected Memory to be initialized")
	}

	if session.Tools == nil {
		t.Fatal("expected ToolRegistry to be initialized")
	}

	state := session.State()

	if state.Running {
		t.Fatal("expected new session to not be running")
	}

	if state.Closed {
		t.Fatal("expected new session to be open")
	}

	if state.HasAudio {
		t.Fatal("expected new session to have no audio")
	}

	if state.HasSTT {
		t.Fatal("expected new session to have no STT worker")
	}

	if state.HasLLM {
		t.Fatal("expected new session to have no LLM")
	}

	if state.HasTTS {
		t.Fatal("expected new session to have no TTS")
	}
}
func TestCallSessionStart2(t *testing.T) {
	call := &SIPCall{
		CallID: "session-start-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	if !session.Running() {
		t.Fatal("expected session to be running")
	}

	if session.Closed() {
		t.Fatal("expected session to be open")
	}

	state := session.State()

	if !state.Running {
		t.Fatal("expected State().Running=true")
	}

	if state.Closed {
		t.Fatal("expected State().Closed=false")
	}
}
func TestCallSessionStartIdempotent(t *testing.T) {
	call := &SIPCall{
		CallID: "session-start-idempotent-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatalf("second Start should be harmless, got: %v", err)
	}

	if !session.Running() {
		t.Fatal("expected session to remain running")
	}
}
func TestCallSessionClose2(t *testing.T) {
	call := &SIPCall{
		CallID: "session-close-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	if session.Running() {
		t.Fatal("expected session to stop running")
	}

	if !session.Closed() {
		t.Fatal("expected session to be closed")
	}

	state := session.State()

	if state.Running {
		t.Fatal("expected State().Running=false")
	}

	if !state.Closed {
		t.Fatal("expected State().Closed=true")
	}
}
func TestCallSessionCloseIdempotent(t *testing.T) {
	call := &SIPCall{
		CallID: "session-close-idempotent-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatalf(
			"second Close should be harmless, got: %v",
			err,
		)
	}

	if !session.Closed() {
		t.Fatal("expected session to remain closed")
	}

	if session.Running() {
		t.Fatal("expected session to remain stopped")
	}
}
func TestCallSessionStartAfterClose(t *testing.T) {
	call := &SIPCall{
		CallID: "session-restart-after-close-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	err = session.Start()

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed after Close, got %v",
			err,
		)
	}

	if session.Running() {
		t.Fatal("expected session to remain stopped")
	}
}
func TestCallSessionNilReceiver(t *testing.T) {
	var session *CallSession

	if err := session.Start(); err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed from nil Start, got %v",
			err,
		)
	}

	if err := session.Close(); err != nil {
		t.Fatalf(
			"expected nil from nil Close, got %v",
			err,
		)
	}

	state := session.State()

	if !state.Closed {
		t.Fatal("expected nil session State().Closed=true")
	}
}
func TestCallSessionConfigureVoiceNilEngine(t *testing.T) {
	call := &SIPCall{
		CallID: "configure-voice-nil-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	err = session.ConfigureVoice(nil)

	if err == nil {
		t.Fatal("expected error for nil TTS engine")
	}
}
func TestCallSessionConfigureVoiceClosed(t *testing.T) {
	call := &SIPCall{
		CallID: "configure-voice-closed-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	tts := NewTTSEngine(nil)

	err = session.ConfigureVoice(tts)

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestCallSessionConfigureVoice(t *testing.T) {
	call := &SIPCall{
		CallID: "configure-voice-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	tts := NewTTSEngine(nil)

	if err := session.ConfigureVoice(tts); err != nil {
		t.Fatal(err)
	}

	if session.TTS != tts {
		t.Fatal("expected TTS engine to be assigned")
	}

	if session.VoiceEngine == nil {
		t.Fatal("expected VoiceEngine to be created")
	}

	state := session.State()

	if !state.HasTTS {
		t.Fatal("expected session to report HasTTS=true")
	}
}
func TestCallSessionConfigureVoiceReplace(t *testing.T) {
	call := &SIPCall{
		CallID: "configure-voice-replace-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	first := NewTTSEngine(nil)
	second := NewTTSEngine(nil)

	if err := session.ConfigureVoice(first); err != nil {
		t.Fatal(err)
	}

	firstVoice := session.VoiceEngine

	if err := session.ConfigureVoice(second); err != nil {
		t.Fatal(err)
	}

	if session.TTS != second {
		t.Fatal("expected TTS engine to be replaced")
	}

	if session.VoiceEngine == nil {
		t.Fatal("expected VoiceEngine to exist")
	}

	if session.VoiceEngine == firstVoice {
		t.Fatal("expected a new VoiceEngine after reconfiguration")
	}
}
func TestCallSessionConfigureVoiceNilReceiver(t *testing.T) {
	var session *CallSession

	tts := NewTTSEngine(nil)

	err := session.ConfigureVoice(tts)

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestCallSessionConfigureSTTNilWorker(t *testing.T) {
	call := &SIPCall{
		CallID: "configure-stt-nil-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	err = session.ConfigureSTT(nil)

	if err == nil {
		t.Fatal("expected error for nil STT worker")
	}
}
func TestCallSessionConfigureSTTClosed(t *testing.T) {
	call := &SIPCall{
		CallID: "configure-stt-closed-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	worker := &STTWorker{}

	err = session.ConfigureSTT(worker)

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestCallSessionConfigureSTT(t *testing.T) {
	call := &SIPCall{
		CallID: "configure-stt-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	worker := &STTWorker{}

	if err := session.ConfigureSTT(worker); err != nil {
		t.Fatal(err)
	}

	if session.STTWorker != worker {
		t.Fatal("expected STT worker to be assigned")
	}

	state := session.State()

	if !state.HasSTT {
		t.Fatal("expected session to report HasSTT=true")
	}

	got := session.GetSTTWorker()

	if got != worker {
		t.Fatal("expected GetSTTWorker to return configured worker")
	}
}
func TestCallSessionConfigureSTTReplace(t *testing.T) {
	call := &SIPCall{
		CallID: "configure-stt-replace-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	first := &STTWorker{}
	second := &STTWorker{}

	if err := session.ConfigureSTT(first); err != nil {
		t.Fatal(err)
	}

	if err := session.ConfigureSTT(second); err != nil {
		t.Fatal(err)
	}

	if session.GetSTTWorker() != second {
		t.Fatal("expected STT worker to be replaced")
	}
}
func TestCallSessionSetSTTWorker2(t *testing.T) {
	call := &SIPCall{
		CallID: "set-stt-worker-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	worker := &STTWorker{}

	if err := session.SetSTTWorker(worker); err != nil {
		t.Fatal(err)
	}

	if session.GetSTTWorker() != worker {
		t.Fatal("expected worker to be assigned")
	}
}
func TestCallSessionSetSTTWorkerNil(t *testing.T) {
	call := &SIPCall{
		CallID: "set-stt-worker-nil-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.SetSTTWorker(nil); err != nil {
		t.Fatal(err)
	}

	if session.GetSTTWorker() != nil {
		t.Fatal("expected STT worker to be nil")
	}
}
func TestCallSessionSetSTTWorkerClosed(t *testing.T) {
	call := &SIPCall{
		CallID: "set-stt-worker-closed-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	worker := &STTWorker{}

	err = session.SetSTTWorker(worker)

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestCallSessionAttachSTTEventsNilReceiver(t *testing.T) {
	var session *CallSession

	if err := session.AttachSTTEvents(); err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestCallSessionAttachSTTEventsClosed(t *testing.T) {
	call := &SIPCall{
		CallID: "attach-stt-events-closed-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	if err := session.AttachSTTEvents(); err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestCallSessionAttachSTTEventsNilBus(t *testing.T) {
	call := &SIPCall{
		CallID: "attach-stt-events-nil-bus-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	session.Events = nil

	err = session.AttachSTTEvents()

	if err == nil {
		t.Fatal("expected error when EventBus is nil")
	}
}
func TestCallSessionAttachSTTEventsWithoutConversationEngine(t *testing.T) {
	call := &SIPCall{
		CallID: "attach-stt-events-no-engine-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	err = session.AttachSTTEvents()

	if err == nil {
		t.Fatal("expected error when ConversationEngine is not configured")
	}
}
func TestCallSessionAttachSTTEvents(t *testing.T) {
	call := &SIPCall{
		CallID: "attach-stt-events-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	llm := NewLLMEngine(nil)

	if err := session.ConfigureConversation(llm); err != nil {
		t.Fatal(err)
	}

	if !session.sttEventsAttached {
		t.Fatal("expected STT events to be attached")
	}
}
func TestCallSessionAttachSTTEventsIdempotent(t *testing.T) {
	call := &SIPCall{
		CallID: "attach-stt-events-idempotent-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	llm := NewLLMEngine(nil)

	if err := session.ConfigureConversation(llm); err != nil {
		t.Fatal(err)
	}

	if !session.sttEventsAttached {
		t.Fatal("expected STT events to be attached")
	}

	if err := session.AttachSTTEvents(); err != nil {
		t.Fatal(err)
	}

	if !session.sttEventsAttached {
		t.Fatal("expected STT events to remain attached")
	}
}
func TestCallSessionConfigureConversationAttachesSTTEvents(t *testing.T) {
	call := &SIPCall{
		CallID: "conversation-stt-events-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if session.sttEventsAttached {
		t.Fatal("expected STT events to be detached initially")
	}

	llm := NewLLMEngine(nil)

	if err := session.ConfigureConversation(llm); err != nil {
		t.Fatal(err)
	}

	if session.ConversationEngine == nil {
		t.Fatal("expected ConversationEngine")
	}

	if !session.sttEventsAttached {
		t.Fatal(
			"expected ConfigureConversation to attach STT events",
		)
	}
}
func TestConversationEngineHandleTranscriptEmptyCallID1(t *testing.T) {
	engine := NewConversationEngine(
		NewLLMEngine(nil),
		NewEventBus(),
	)

	err := engine.HandleTranscript(TranscriptEvent{})

	if err == nil {
		t.Fatal("expected error for empty CallID")
	}
}
func TestConversationEngineHandleTranscriptEmptyText2(t *testing.T) {
	engine := NewConversationEngine(
		NewLLMEngine(nil),
		NewEventBus(),
	)

	event := TranscriptEvent{
		CallID: "empty-text-call",
	}

	err := engine.HandleTranscript(event)

	if err != nil {
		t.Fatalf(
			"expected empty transcript to be ignored, got %v",
			err,
		)
	}

	if _, err := engine.GetOrCreate("empty-text-call"); err != nil {
		t.Fatal(err)
	}
}
func TestConversationEngineGetOrCreateSameConversation(t *testing.T) {
	engine := NewConversationEngine(
		NewLLMEngine(nil),
		NewEventBus(),
	)

	first, err := engine.GetOrCreate("same-conversation-call")
	if err != nil {
		t.Fatal(err)
	}

	second, err := engine.GetOrCreate("same-conversation-call")
	if err != nil {
		t.Fatal(err)
	}

	if first != second {
		t.Fatal("expected GetOrCreate to return the same conversation")
	}
}
func TestConversationEngineHandleTranscriptAddsUserMessage(t *testing.T) {
	engine := NewConversationEngine(
		NewLLMEngine(nil),
		NewEventBus(),
	)

	event := TranscriptEvent{
		CallID: "transcript-message-test",
	}

	event.Transcript.Text = "hello neurocall"

	err := engine.HandleTranscript(event)

	if err == nil {
		t.Fatal("expected error because LLM is not configured")
	}

	conversation, err := engine.GetOrCreate("transcript-message-test")
	if err != nil {
		t.Fatal(err)
	}

	messages := conversation.Messages()

	if len(messages) != 1 {
		t.Fatalf(
			"expected 1 message, got %d",
			len(messages),
		)
	}

	if messages[0].Role != "user" {
		t.Fatalf(
			"expected user role, got %q",
			messages[0].Role,
		)
	}

	if messages[0].Content != "hello neurocall" {
		t.Fatalf(
			"expected transcript text %q, got %q",
			"hello neurocall",
			messages[0].Content,
		)
	}
}
func TestCallSessionAttachSTTEventsFiltersCallID(t *testing.T) {
	call := &SIPCall{
		CallID: "session-transcript-filter-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	llm := NewLLMEngine(nil)

	if err := session.ConfigureConversation(llm); err != nil {
		t.Fatal(err)
	}

	engine := session.ConversationEngine

	session.Events.Publish(Event{
		Name: EventTranscript,
		Data: TranscriptEvent{
			CallID: "another-call",
			Transcript: neurocall.Transcript{
				Text: "should be ignored",
			},
		},
	})

	time.Sleep(100 * time.Millisecond)

	otherConversation, err := engine.GetOrCreate("another-call")
	if err != nil {
		t.Fatal(err)
	}

	if messages := otherConversation.Messages(); len(messages) != 0 {
		t.Fatalf(
			"expected wrong CallID to be ignored, got %d messages",
			len(messages),
		)
	}

	session.Events.Publish(Event{
		Name: EventTranscript,
		Data: TranscriptEvent{
			CallID: session.ID,
			Transcript: neurocall.Transcript{
				Text: "hello neurocall",
			},
		},
	})

	var conversation *Conversation

	deadline := time.Now().Add(1 * time.Second)

	for time.Now().Before(deadline) {
		conversation, err = engine.GetOrCreate(session.ID)
		if err != nil {
			t.Fatal(err)
		}

		if len(conversation.Messages()) > 0 {
			break
		}

		time.Sleep(10 * time.Millisecond)
	}

	if conversation == nil {
		t.Fatal("expected conversation to exist")
	}

	messages := conversation.Messages()

	if len(messages) != 1 {
		t.Fatalf(
			"expected 1 user message, got %d",
			len(messages),
		)
	}

	if messages[0].Role != "user" {
		t.Fatalf(
			"expected user role, got %q",
			messages[0].Role,
		)
	}

	if messages[0].Content != "hello neurocall" {
		t.Fatalf(
			"expected transcript text %q, got %q",
			"hello neurocall",
			messages[0].Content,
		)
	}
}
func TestCallSessionAttachSTTEventsIgnoresInvalidEventData(t *testing.T) {
	call := &SIPCall{
		CallID: "invalid-transcript-event-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	llm := NewLLMEngine(nil)

	if err := session.ConfigureConversation(llm); err != nil {
		t.Fatal(err)
	}

	errorCh := make(chan struct{}, 1)

	session.Events.Subscribe(
		EventLLMError,
		func(event Event) {
			errorCh <- struct{}{}
		},
	)

	session.Events.Publish(Event{
		Name: EventTranscript,
		Data: "this is not a TranscriptEvent",
	})

	select {
	case <-errorCh:
		t.Fatal("expected invalid event data to be ignored")

	case <-time.After(100 * time.Millisecond):
	}
}
func TestNewSTTWorkerNilEngine(t *testing.T) {
	bus := NewEventBus()

	worker, err := NewSTTWorker(
		nil,
		bus,
		"constructor-test",
		10,
	)

	if err == nil {
		t.Fatal("expected error for nil STT engine")
	}

	if worker != nil {
		t.Fatal("expected nil worker")
	}
}
func TestNewSTTWorkerNilBus(t *testing.T) {
	engine := NewSTTEngine(nil)

	worker, err := NewSTTWorker(
		engine,
		nil,
		"constructor-test",
		10,
	)

	if err == nil {
		t.Fatal("expected error for nil EventBus")
	}

	if worker != nil {
		t.Fatal("expected nil worker")
	}
}
func TestNewSTTWorkerEmptyCallID(t *testing.T) {
	engine := NewSTTEngine(nil)
	bus := NewEventBus()

	worker, err := NewSTTWorker(
		engine,
		bus,
		"",
		10,
	)

	if err == nil {
		t.Fatal("expected error for empty CallID")
	}

	if worker != nil {
		t.Fatal("expected nil worker")
	}
}
func TestNewSTTWorkerDefaultBufferSize(t *testing.T) {
	engine := NewSTTEngine(nil)
	bus := NewEventBus()

	worker, err := NewSTTWorker(
		engine,
		bus,
		"buffer-default-test",
		0,
	)

	if err != nil {
		t.Fatal(err)
	}

	if worker == nil {
		t.Fatal("expected worker")
	}

	if cap(worker.input) != 16 {
		t.Fatalf(
			"expected default buffer size 16, got %d",
			cap(worker.input),
		)
	}
}
func TestSTTWorkerPushInvalidAudio(t *testing.T) {
	engine := NewSTTEngine(nil)
	bus := NewEventBus()

	worker, err := NewSTTWorker(
		engine,
		bus,
		"push-invalid-test",
		10,
	)

	if err != nil {
		t.Fatal(err)
	}

	err = worker.Push(neurocall.AudioSegment{})

	if err != neurocall.ErrInvalidAudio {
		t.Fatalf(
			"expected ErrInvalidAudio, got %v",
			err,
		)
	}
}
func TestSTTWorkerPushBeforeStart(t *testing.T) {
	engine := NewSTTEngine(nil)
	bus := NewEventBus()

	worker, err := NewSTTWorker(
		engine,
		bus,
		"push-before-start-test",
		10,
	)

	if err != nil {
		t.Fatal(err)
	}

	err = worker.Push(neurocall.AudioSegment{
		Data: []int16{1, 2, 3},
	})

	if err != nil {
		t.Fatalf(
			"expected Push before Start to succeed, got %v",
			err,
		)
	}
}
func TestSTTWorkerStart(t *testing.T) {
	engine := NewSTTEngine(nil)
	bus := NewEventBus()

	worker, err := NewSTTWorker(
		engine,
		bus,
		"start-test",
		10,
	)

	if err != nil {
		t.Fatal(err)
	}

	if err := worker.Start(); err != nil {
		t.Fatal(err)
	}

	worker.mu.Lock()
	running := worker.running
	stopped := worker.stopped
	worker.mu.Unlock()

	if !running {
		t.Fatal("expected worker to be running")
	}

	if stopped {
		t.Fatal("expected worker to not be stopped")
	}
}
func TestSTTWorkerStartIdempotent(t *testing.T) {
	engine := NewSTTEngine(nil)
	bus := NewEventBus()

	worker, err := NewSTTWorker(
		engine,
		bus,
		"start-idempotent-test",
		10,
	)

	if err != nil {
		t.Fatal(err)
	}

	if err := worker.Start(); err != nil {
		t.Fatal(err)
	}

	if err := worker.Start(); err != nil {
		t.Fatalf(
			"second Start should be harmless, got %v",
			err,
		)
	}

	worker.mu.Lock()
	running := worker.running
	worker.mu.Unlock()

	if !running {
		t.Fatal("expected worker to remain running")
	}
}
func TestSTTWorkerStop(t *testing.T) {
	engine := NewSTTEngine(nil)
	bus := NewEventBus()

	worker, err := NewSTTWorker(
		engine,
		bus,
		"stop-test",
		10,
	)

	if err != nil {
		t.Fatal(err)
	}

	if err := worker.Start(); err != nil {
		t.Fatal(err)
	}

	worker.Stop()

	worker.mu.Lock()
	running := worker.running
	stopped := worker.stopped
	worker.mu.Unlock()

	if running {
		t.Fatal("expected worker to stop running")
	}

	if !stopped {
		t.Fatal("expected worker to be marked stopped")
	}
}
func TestSTTWorkerStopIdempotent(t *testing.T) {
	engine := NewSTTEngine(nil)
	bus := NewEventBus()

	worker, err := NewSTTWorker(
		engine,
		bus,
		"stop-idempotent-test",
		10,
	)

	if err != nil {
		t.Fatal(err)
	}

	if err := worker.Start(); err != nil {
		t.Fatal(err)
	}

	worker.Stop()
	worker.Stop()

	worker.mu.Lock()
	stopped := worker.stopped
	running := worker.running
	worker.mu.Unlock()

	if !stopped {
		t.Fatal("expected worker to remain stopped")
	}

	if running {
		t.Fatal("expected worker to remain not running")
	}
}
func TestSTTWorkerStartAfterStop(t *testing.T) {
	engine := NewSTTEngine(nil)
	bus := NewEventBus()

	worker, err := NewSTTWorker(
		engine,
		bus,
		"restart-after-stop-test",
		10,
	)

	if err != nil {
		t.Fatal(err)
	}

	worker.Stop()

	err = worker.Start()

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestSTTWorkerPushAfterStop(t *testing.T) {
	engine := NewSTTEngine(nil)
	bus := NewEventBus()

	worker, err := NewSTTWorker(
		engine,
		bus,
		"push-after-stop-test",
		10,
	)

	if err != nil {
		t.Fatal(err)
	}

	worker.Stop()

	err = worker.Push(neurocall.AudioSegment{
		Data: []int16{1, 2, 3},
	})

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestSTTWorkerEmitsTranscript(t *testing.T) {
	transcriptCh := make(chan TranscriptEvent, 1)

	stt := &testSTT{
		transcribe: func(
			ctx context.Context,
			segment neurocall.AudioSegment,
		) (neurocall.Transcript, error) {
			if len(segment.Data) == 0 {
				t.Fatal("expected audio data")
			}

			return neurocall.Transcript{}, nil
		},
	}

	bus := NewEventBus()

	bus.Subscribe(
		EventTranscript,
		func(event Event) {
			data, ok := event.Data.(TranscriptEvent)
			if !ok {
				t.Errorf("expected TranscriptEvent, got %T", event.Data)
				return
			}

			transcriptCh <- data
		},
	)

	engine := NewSTTEngine(stt)

	worker, err := NewSTTWorker(
		engine,
		bus,
		"stt-transcript-test",
		10,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := worker.Start(); err != nil {
		t.Fatal(err)
	}

	defer worker.Stop()

	err = worker.Push(neurocall.AudioSegment{
		Data: []int16{1, 2, 3, 4},
	})
	if err != nil {
		t.Fatal(err)
	}

	select {
	case event := <-transcriptCh:
		if event.CallID != "stt-transcript-test" {
			t.Fatalf(
				"expected CallID %q, got %q",
				"stt-transcript-test",
				event.CallID,
			)
		}

	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for EventTranscript")
	}
}
func TestSTTWorkerEmitsSTTError(t *testing.T) {
	expectedErr := errors.New("transcription failed")

	errorCh := make(chan STTErrorEvent, 1)

	stt := &testSTT{
		transcribe: func(
			ctx context.Context,
			segment neurocall.AudioSegment,
		) (neurocall.Transcript, error) {
			return neurocall.Transcript{}, expectedErr
		},
	}

	bus := NewEventBus()

	bus.Subscribe(
		EventSTTError,
		func(event Event) {
			data, ok := event.Data.(STTErrorEvent)
			if !ok {
				t.Errorf("expected STTErrorEvent, got %T", event.Data)
				return
			}

			errorCh <- data
		},
	)

	engine := NewSTTEngine(stt)

	worker, err := NewSTTWorker(
		engine,
		bus,
		"stt-error-test",
		10,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := worker.Start(); err != nil {
		t.Fatal(err)
	}

	defer worker.Stop()

	if err := worker.Push(neurocall.AudioSegment{
		Data: []int16{1, 2, 3},
	}); err != nil {
		t.Fatal(err)
	}

	select {
	case event := <-errorCh:
		if event.CallID != "stt-error-test" {
			t.Fatalf(
				"expected CallID %q, got %q",
				"stt-error-test",
				event.CallID,
			)
		}

		if event.Err != expectedErr {
			t.Fatalf(
				"expected error %v, got %v",
				expectedErr,
				event.Err,
			)
		}

	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for EventSTTError")
	}
}
func TestSTTEngineTranscribeWithoutSTT(t *testing.T) {
	engine := NewSTTEngine(nil)

	_, err := engine.Transcribe(
		context.Background(),
		neurocall.AudioSegment{
			Data: []int16{1, 2, 3},
		},
	)

	if err == nil {
		t.Fatal("expected error when STT is not configured")
	}
}
func TestSTTEngineCloseIdempotent(t *testing.T) {
	engine := NewSTTEngine(nil)

	engine.Close()
	engine.Close()
}
func TestCallSessionSTTToConversationIntegration(t *testing.T) {
	call := &SIPCall{
		CallID: "stt-conversation-integration-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	llm := NewLLMEngine(nil)

	if err := session.ConfigureConversation(llm); err != nil {
		t.Fatal(err)
	}

	stt := &testSTT{
		transcribe: func(
			ctx context.Context,
			segment neurocall.AudioSegment,
		) (neurocall.Transcript, error) {
			return neurocall.Transcript{
				Text: "hello from STT",
			}, nil
		},
	}

	sttEngine := NewSTTEngine(stt)

	worker, err := NewSTTWorker(
		sttEngine,
		session.Events,
		session.ID,
		10,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.ConfigureSTT(worker); err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	defer session.Close()

	if err := worker.Start(); err != nil {
		t.Fatal(err)
	}

	defer worker.Stop()

	if err := worker.Push(neurocall.AudioSegment{
		Data: []int16{1, 2, 3, 4},
	}); err != nil {
		t.Fatal(err)
	}

	engine := session.ConversationEngine

	var conversation *Conversation

	deadline := time.Now().Add(1 * time.Second)

	for time.Now().Before(deadline) {
		conversation, err = engine.GetOrCreate(session.ID)
		if err != nil {
			t.Fatal(err)
		}

		messages := conversation.Messages()

		if len(messages) > 0 {
			break
		}

		time.Sleep(10 * time.Millisecond)
	}

	if conversation == nil {
		t.Fatal("expected conversation to exist")
	}

	messages := conversation.Messages()

	if len(messages) != 1 {
		t.Fatalf(
			"expected 1 conversation message, got %d",
			len(messages),
		)
	}

	if messages[0].Role != "user" {
		t.Fatalf(
			"expected user role, got %q",
			messages[0].Role,
		)
	}

	if messages[0].Content != "hello from STT" {
		t.Fatalf(
			"expected transcript text %q, got %q",
			"hello from STT",
			messages[0].Content,
		)
	}
}
func TestCallSessionSTTToConversationIntegration1(t *testing.T) {
	call := &SIPCall{
		CallID: "stt-conversation-integration-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	llm := NewLLMEngine(nil)

	if err := session.ConfigureConversation(llm); err != nil {
		t.Fatal(err)
	}

	stt := &testSTT{
		transcribe: func(
			ctx context.Context,
			segment neurocall.AudioSegment,
		) (neurocall.Transcript, error) {
			return neurocall.Transcript{
				Text: "hello from STT",
			}, nil
		},
	}

	sttEngine := NewSTTEngine(stt)

	worker, err := NewSTTWorker(
		sttEngine,
		session.Events,
		session.ID,
		10,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.ConfigureSTT(worker); err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	defer session.Close()

	if err := worker.Push(neurocall.AudioSegment{
		Data: []int16{1, 2, 3, 4},
	}); err != nil {
		t.Fatal(err)
	}

	engine := session.ConversationEngine

	var conversation *Conversation

	deadline := time.Now().Add(1 * time.Second)

	for time.Now().Before(deadline) {
		conversation, err = engine.GetOrCreate(session.ID)
		if err != nil {
			t.Fatal(err)
		}

		if len(conversation.Messages()) > 0 {
			break
		}

		time.Sleep(10 * time.Millisecond)
	}

	if conversation == nil {
		t.Fatal("expected conversation to exist")
	}

	messages := conversation.Messages()

	if len(messages) != 1 {
		t.Fatalf(
			"expected 1 conversation message, got %d",
			len(messages),
		)
	}

	if messages[0].Role != "user" {
		t.Fatalf(
			"expected user role, got %q",
			messages[0].Role,
		)
	}

	if messages[0].Content != "hello from STT" {
		t.Fatalf(
			"expected transcript text %q, got %q",
			"hello from STT",
			messages[0].Content,
		)
	}
}
func TestCallSessionProcessAudioSegmentBeforeStart(t *testing.T) {
	call := &SIPCall{
		CallID: "process-segment-before-start-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	worker := &STTWorker{}

	if err := session.ConfigureSTT(worker); err != nil {
		t.Fatal(err)
	}

	err = session.ProcessAudioSegment(neurocall.AudioSegment{
		Data: []int16{1, 2, 3},
	})

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestCallSessionProcessAudioSegmentInvalidAudio(t *testing.T) {
	call := &SIPCall{
		CallID: "process-segment-invalid-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	err = session.ProcessAudioSegment(
		neurocall.AudioSegment{},
	)

	if err != neurocall.ErrInvalidAudio {
		t.Fatalf(
			"expected ErrInvalidAudio, got %v",
			err,
		)
	}
}
func TestCallSessionProcessAudioSegmentWithoutSTTWorker(t *testing.T) {
	call := &SIPCall{
		CallID: "process-segment-no-worker-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	defer session.Close()

	err = session.ProcessAudioSegment(
		neurocall.AudioSegment{
			Data: []int16{1, 2, 3},
		},
	)

	if err == nil {
		t.Fatal("expected error when STT worker is not configured")
	}
}
func TestCallSessionProcessAudioSegment(t *testing.T) {
	call := &SIPCall{
		CallID: "process-segment-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	stt := &testSTT{
		transcribe: func(
			ctx context.Context,
			segment neurocall.AudioSegment,
		) (neurocall.Transcript, error) {
			return neurocall.Transcript{
				Text: "processed segment",
			}, nil
		},
	}

	engine := NewSTTEngine(stt)

	worker, err := NewSTTWorker(
		engine,
		session.Events,
		session.ID,
		10,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.ConfigureSTT(worker); err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	defer session.Close()

	transcriptCh := make(chan TranscriptEvent, 1)

	session.Events.Subscribe(
		EventTranscript,
		func(event Event) {
			data, ok := event.Data.(TranscriptEvent)
			if ok {
				transcriptCh <- data
			}
		},
	)

	err = session.ProcessAudioSegment(
		neurocall.AudioSegment{
			Data: []int16{1, 2, 3, 4},
		},
	)

	if err != nil {
		t.Fatal(err)
	}

	select {
	case transcript := <-transcriptCh:
		if transcript.CallID != session.ID {
			t.Fatalf(
				"expected CallID %q, got %q",
				session.ID,
				transcript.CallID,
			)
		}

		if transcript.Transcript.Text != "processed segment" {
			t.Fatalf(
				"expected transcript %q, got %q",
				"processed segment",
				transcript.Transcript.Text,
			)
		}

	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for transcript")
	}
}
func TestCallSessionProcessAudioSegmentAfterClose(t *testing.T) {
	call := &SIPCall{
		CallID: "process-segment-after-close-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	err = session.ProcessAudioSegment(
		neurocall.AudioSegment{
			Data: []int16{1, 2, 3},
		},
	)

	if err != neurocall.ErrCallClosed {
		t.Fatalf(
			"expected ErrCallClosed, got %v",
			err,
		)
	}
}
func TestCallSessionProcessAudioFrameInvalidAudio(t *testing.T) {
	call := &SIPCall{
		CallID: "process-frame-invalid-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	err = session.ProcessAudioFrame(
		context.Background(),
		neurocall.AudioFrame{},
		[]int16{},
	)

	if err != neurocall.ErrInvalidAudio {
		t.Fatalf(
			"expected ErrInvalidAudio, got %v",
			err,
		)
	}
}
func TestCallSessionProcessAudioFrameWithoutSegmenter(t *testing.T) {
	call := &SIPCall{
		CallID: "process-frame-no-segmenter-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	defer session.Close()

	err = session.ProcessAudioFrame(
		context.Background(),
		neurocall.AudioFrame{},
		[]int16{1000, 1000},
	)

	if err == nil {
		t.Fatal("expected error when segmenter is not configured")
	}
}
func TestCallSessionProcessAudioFrameWithoutSTTWorker(t *testing.T) {
	call := &SIPCall{
		CallID: "process-frame-no-worker-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	vad := NewEnergyVAD(500)

	segmenter := NewAudioSegmenter(
		vad,
		AudioConfig{
			SampleRate: 8000,
			FrameSize:  160,
			Channels:   1,
		},
	)

	call.VAD = vad
	call.Segmenter = segmenter

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	defer session.Close()

	err = session.ProcessAudioFrame(
		context.Background(),
		neurocall.AudioFrame{
			Data:      []int16{1000, 1000, 1000},
			Timestamp: 0,
		},
		[]int16{1000, 1000, 1000},
	)

	if err == nil {
		t.Fatal("expected error when STT worker is not configured")
	}
}
func TestCallSessionProcessAudioFrameSilence(t *testing.T) {
	call := &SIPCall{
		CallID: "process-frame-silence-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	vad := NewEnergyVAD(500)

	segmenter := NewAudioSegmenter(
		vad,
		AudioConfig{
			SampleRate: 8000,
			FrameSize:  160,
			Channels:   1,
		},
	)

	call.VAD = vad
	call.Segmenter = segmenter

	stt := &testSTT{
		transcribe: func(
			ctx context.Context,
			segment neurocall.AudioSegment,
		) (neurocall.Transcript, error) {
			return neurocall.Transcript{
				Text: "should not happen",
			}, nil
		},
	}

	engine := NewSTTEngine(stt)

	worker, err := NewSTTWorker(
		engine,
		session.Events,
		session.ID,
		10,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.ConfigureSTT(worker); err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	defer session.Close()

	transcriptCh := make(chan TranscriptEvent, 1)

	session.Events.Subscribe(
		EventTranscript,
		func(event Event) {
			if data, ok := event.Data.(TranscriptEvent); ok {
				transcriptCh <- data
			}
		},
	)

	err = session.ProcessAudioFrame(
		context.Background(),
		neurocall.AudioFrame{
			Data:      []int16{10, 10, 10, 10},
			Timestamp: 0,
		},
		[]int16{10, 10, 10, 10},
	)

	if err != nil {
		t.Fatal(err)
	}

	select {
	case event := <-transcriptCh:
		t.Fatalf(
			"unexpected transcript: %+v",
			event,
		)

	case <-time.After(100 * time.Millisecond):
	}
}
func TestCallSessionProcessAudioFrameSpeechProducesTranscript(t *testing.T) {
	call := &SIPCall{
		CallID: "process-frame-speech-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	vad := NewEnergyVAD(500)

	segmenter := NewAudioSegmenter(
		vad,
		AudioConfig{
			SampleRate: 8000,
			FrameSize:  160,
			Channels:   1,
		},
	)

	call.VAD = vad
	call.Segmenter = segmenter

	stt := &testSTT{
		transcribe: func(
			ctx context.Context,
			segment neurocall.AudioSegment,
		) (neurocall.Transcript, error) {
			if len(segment.Data) == 0 {
				t.Fatal("expected segment data")
			}

			return neurocall.Transcript{
				Text: "speech detected",
			}, nil
		},
	}

	engine := NewSTTEngine(stt)

	worker, err := NewSTTWorker(
		engine,
		session.Events,
		session.ID,
		10,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.ConfigureSTT(worker); err != nil {
		t.Fatal(err)
	}

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	defer session.Close()

	transcriptCh := make(chan TranscriptEvent, 1)

	session.Events.Subscribe(
		EventTranscript,
		func(event Event) {
			if data, ok := event.Data.(TranscriptEvent); ok {
				transcriptCh <- data
			}
		},
	)

	speechFrame := neurocall.AudioFrame{
		Data:      []int16{1000, 1000, 1000, 1000},
		Timestamp: 0,
	}

	for i := 0; i < 2; i++ {
		err := session.ProcessAudioFrame(
			context.Background(),
			speechFrame,
			speechFrame.Data,
		)

		if err != nil {
			t.Fatal(err)
		}
	}

	silenceFrame := neurocall.AudioFrame{
		Data:      []int16{0, 0, 0, 0},
		Timestamp: 20,
	}

	for i := 0; i < 10; i++ {
		err := session.ProcessAudioFrame(
			context.Background(),
			silenceFrame,
			silenceFrame.Data,
		)

		if err != nil {
			t.Fatal(err)
		}
	}

	select {
	case event := <-transcriptCh:
		if event.CallID != session.ID {
			t.Fatalf(
				"expected CallID %q, got %q",
				session.ID,
				event.CallID,
			)
		}

		if event.Transcript.Text != "speech detected" {
			t.Fatalf(
				"expected transcript %q, got %q",
				"speech detected",
				event.Transcript.Text,
			)
		}

	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for transcript")
	}
}
func TestCallSessionProcessAudioFrameCancelledContext(t *testing.T) {
	call := &SIPCall{
		CallID: "process-frame-context-test",
	}

	session, err := NewCallSession(call, NewEventBus())
	if err != nil {
		t.Fatal(err)
	}

	vad := NewEnergyVAD(500)

	call.VAD = vad

	call.Segmenter = NewAudioSegmenter(
		vad,
		AudioConfig{
			SampleRate: 8000,
			FrameSize:  160,
			Channels:   1,
		},
	)

	if err := session.Start(); err != nil {
		t.Fatal(err)
	}

	defer session.Close()

	ctx, cancel := context.WithCancel(
		context.Background(),
	)
	cancel()

	err = session.ProcessAudioFrame(
		ctx,
		neurocall.AudioFrame{
			Data: []int16{1000, 1000},
		},
		[]int16{1000, 1000},
	)

	if err == nil {
		t.Fatal("expected error")
	}

	if !errors.Is(err, context.Canceled) {
		t.Fatalf(
			"expected context.Canceled, got %v",
			err,
		)
	}
}
func TestRTPSequenceTrackerForward(t *testing.T) {
	tracker := NewRTPSequenceTracker()

	tracker.Update(10)
	tracker.Update(11)
	tracker.Update(12)

	stats := tracker.Stats()

	if stats.ReceivedPackets != 3 {
		t.Fatalf(
			"expected 3 received, got %d",
			stats.ReceivedPackets,
		)
	}

	if stats.LostPackets != 0 {
		t.Fatalf(
			"expected 0 lost, got %d",
			stats.LostPackets,
		)
	}
}
func TestRTPSequenceTrackerLoss(t *testing.T) {
	tracker := NewRTPSequenceTracker()

	tracker.Update(10)
	tracker.Update(13)

	stats := tracker.Stats()

	if stats.ReceivedPackets != 2 {
		t.Fatalf(
			"expected 2 received, got %d",
			stats.ReceivedPackets,
		)
	}

	if stats.LostPackets != 2 {
		t.Fatalf(
			"expected 2 lost, got %d",
			stats.LostPackets,
		)
	}
}
func TestRTPSequenceTrackerWrapAround(t *testing.T) {
	tracker := NewRTPSequenceTracker()

	tracker.Update(65534)
	tracker.Update(65535)
	tracker.Update(0)
	tracker.Update(1)

	stats := tracker.Stats()

	if stats.ReceivedPackets != 4 {
		t.Fatalf(
			"expected 4 received, got %d",
			stats.ReceivedPackets,
		)
	}

	if stats.LostPackets != 0 {
		t.Fatalf(
			"expected 0 lost, got %d",
			stats.LostPackets,
		)
	}

	if stats.LastSequence != 1 {
		t.Fatalf(
			"expected last sequence 1, got %d",
			stats.LastSequence,
		)
	}
}
func TestRTPSequenceTrackerWrapAroundLoss(t *testing.T) {
	tracker := NewRTPSequenceTracker()

	tracker.Update(65534)
	tracker.Update(1)

	stats := tracker.Stats()

	if stats.ReceivedPackets != 2 {
		t.Fatalf(
			"expected 2 received, got %d",
			stats.ReceivedPackets,
		)
	}

	if stats.LostPackets != 2 {
		t.Fatalf(
			"expected 2 lost, got %d",
			stats.LostPackets,
		)
	}
}
func TestRTPSequenceTrackerDuplicate1(t *testing.T) {
	tracker := NewRTPSequenceTracker()

	tracker.Update(100)
	tracker.Update(101)
	tracker.Update(101)

	stats := tracker.Stats()

	if stats.ReceivedPackets != 2 {
		t.Fatalf(
			"expected 2 received, got %d",
			stats.ReceivedPackets,
		)
	}

	if stats.Duplicates != 1 {
		t.Fatalf(
			"expected 1 duplicate, got %d",
			stats.Duplicates,
		)
	}
}
func TestRTPSequenceTrackerOutOfOrder1(t *testing.T) {
	tracker := NewRTPSequenceTracker()

	tracker.Update(100)
	tracker.Update(102)
	tracker.Update(101)

	stats := tracker.Stats()

	if stats.ReceivedPackets != 2 {
		t.Fatalf(
			"expected 2 received, got %d",
			stats.ReceivedPackets,
		)
	}

	if stats.LostPackets != 1 {
		t.Fatalf(
			"expected 1 lost, got %d",
			stats.LostPackets,
		)
	}

	if stats.OutOfOrder != 1 {
		t.Fatalf(
			"expected 1 out-of-order, got %d",
			stats.OutOfOrder,
		)
	}
}
func TestSeqLess(t *testing.T) {
	tests := []struct {
		name string
		a    uint16
		b    uint16
		want bool
	}{
		{
			name: "normal forward",
			a:    10,
			b:    11,
			want: true,
		},
		{
			name: "normal backward",
			a:    11,
			b:    10,
			want: false,
		},
		{
			name: "equal",
			a:    10,
			b:    10,
			want: false,
		},
		{
			name: "wrap",
			a:    65535,
			b:    0,
			want: true,
		},
		{
			name: "after wrap",
			a:    0,
			b:    1,
			want: true,
		},
		{
			name: "before wrap",
			a:    65534,
			b:    65535,
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := seqLess(tt.a, tt.b)

			if got != tt.want {
				t.Fatalf(
					"seqLess(%d, %d) = %v, want %v",
					tt.a,
					tt.b,
					got,
					tt.want,
				)
			}
		})
	}
}
func TestRTPJitterBufferOrdered(t *testing.T) {
	b := NewRTPJitterBuffer(
		10,
		30*time.Millisecond,
	)

	for _, seq := range []uint16{100, 101, 102} {
		err := b.Push(&RTPPacket{
			Header: RTPHeader{
				SequenceNumber: seq,
			},
		})

		if err != nil {
			t.Fatal(err)
		}
	}

	for _, expected := range []uint16{100, 101, 102} {
		packet, ready := b.Pop()

		if !ready {
			t.Fatalf(
				"expected packet %d to be ready",
				expected,
			)
		}

		if packet == nil {
			t.Fatalf(
				"expected packet %d, got nil",
				expected,
			)
		}

		if packet.Header.SequenceNumber != expected {
			t.Fatalf(
				"expected %d, got %d",
				expected,
				packet.Header.SequenceNumber,
			)
		}
	}
}
func TestRTPJitterBufferReordersPackets(t *testing.T) {
	b := NewRTPJitterBuffer(
		10,
		100*time.Millisecond,
	)

	for _, seq := range []uint16{100, 102, 101} {
		err := b.Push(&RTPPacket{
			Header: RTPHeader{
				SequenceNumber: seq,
			},
		})

		if err != nil {
			t.Fatal(err)
		}
	}

	for _, expected := range []uint16{100, 101, 102} {
		packet, ready := b.Pop()

		if !ready {
			t.Fatalf(
				"expected %d to be ready",
				expected,
			)
		}

		if packet == nil {
			t.Fatalf(
				"expected %d, got nil",
				expected,
			)
		}

		if packet.Header.SequenceNumber != expected {
			t.Fatalf(
				"expected %d, got %d",
				expected,
				packet.Header.SequenceNumber,
			)
		}
	}
}
func TestRTPJitterBufferWrapAround(t *testing.T) {
	b := NewRTPJitterBuffer(
		10,
		100*time.Millisecond,
	)

	for _, seq := range []uint16{
		65534,
		65535,
		0,
		1,
	} {
		err := b.Push(&RTPPacket{
			Header: RTPHeader{
				SequenceNumber: seq,
			},
		})

		if err != nil {
			t.Fatal(err)
		}
	}

	expected := []uint16{
		65534,
		65535,
		0,
		1,
	}

	for _, want := range expected {
		packet, ready := b.Pop()

		if !ready {
			t.Fatalf(
				"expected %d to be ready",
				want,
			)
		}

		if packet == nil {
			t.Fatalf(
				"expected %d, got nil",
				want,
			)
		}

		if packet.Header.SequenceNumber != want {
			t.Fatalf(
				"expected %d, got %d",
				want,
				packet.Header.SequenceNumber,
			)
		}
	}
}
func TestRTPJitterBufferOverflow(t *testing.T) {
	b := NewRTPJitterBuffer(
		3,
		100*time.Millisecond,
	)

	for _, seq := range []uint16{100, 101, 102, 103} {
		if err := b.Push(&RTPPacket{
			Header: RTPHeader{
				SequenceNumber: seq,
			},
		}); err != nil {
			t.Fatal(err)
		}
	}

	if b.HasPackets() == false {
		t.Fatal("expected packets after overflow")
	}

	expected := []uint16{101, 102, 103}

	for _, want := range expected {
		packet, ready := b.Pop()

		if !ready {
			t.Fatalf("expected packet %d to be ready", want)
		}

		if packet == nil {
			t.Fatalf("expected packet %d, got nil", want)
		}

		if packet.Header.SequenceNumber != want {
			t.Fatalf(
				"expected %d, got %d",
				want,
				packet.Header.SequenceNumber,
			)
		}
	}
}
func TestRTPJitterBufferOverflowWrapAround(t *testing.T) {
	b := NewRTPJitterBuffer(
		3,
		100*time.Millisecond,
	)

	for _, seq := range []uint16{
		65534,
		65535,
		0,
		1,
	} {
		if err := b.Push(&RTPPacket{
			Header: RTPHeader{
				SequenceNumber: seq,
			},
		}); err != nil {
			t.Fatal(err)
		}
	}

	expected := []uint16{
		65535,
		0,
		1,
	}

	for _, want := range expected {
		packet, ready := b.Pop()

		if !ready {
			t.Fatalf(
				"expected packet %d to be ready",
				want,
			)
		}

		if packet == nil {
			t.Fatalf(
				"expected packet %d, got nil",
				want,
			)
		}

		if packet.Header.SequenceNumber != want {
			t.Fatalf(
				"expected %d, got %d",
				want,
				packet.Header.SequenceNumber,
			)
		}
	}
}
func TestRTPSessionReadClose(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}

	result := make(chan error, 1)

	go func() {
		_, err := session.Read()
		result <- err
	}()

	time.Sleep(20 * time.Millisecond)

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	select {
	case err := <-result:
		if !errors.Is(err, errRTPSessionClosed) {
			t.Fatalf(
				"expected errRTPSessionClosed, got %v",
				err,
			)
		}

	case <-time.After(time.Second):
		t.Fatal("Read did not unblock after Close")
	}
}
func TestRTPSessionWriteClose(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}

	payload := []byte{0x01, 0x02, 0x03}

	result := make(chan error, 1)

	go func() {
		result <- session.Write(payload, 160)
	}()

	time.Sleep(1 * time.Millisecond)

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	select {
	case err := <-result:
		// Both outcomes are acceptable:
		// the write may win the race or Close may win it.
		if err != nil && !errors.Is(err, errRTPSessionClosed) {
			t.Fatalf("unexpected Write error: %v", err)
		}

	case <-time.After(time.Second):
		t.Fatal("Write did not return")
	}
}
func TestRTPSessionReadOrderedClose(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}

	result := make(chan error, 1)

	go func() {
		_, err := session.ReadOrdered()
		result <- err
	}()

	// Give ReadOrdered enough time to reach blocking Read().
	time.Sleep(20 * time.Millisecond)

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	select {
	case err := <-result:
		if !errors.Is(err, errRTPSessionClosed) {
			t.Fatalf(
				"expected errRTPSessionClosed, got %v",
				err,
			)
		}

	case <-time.After(time.Second):
		t.Fatal("ReadOrdered did not unblock after Close")
	}
}
func TestRTPSessionConcurrentWrite(t *testing.T) {
	codec := neurocall.Codec{
		Name:        "PCMU",
		PayloadType: 0,
		ClockRate:   8000,
		Channels:    1,
	}

	session, err := NewRTPSession(
		"127.0.0.1",
		0,
		net.ParseIP("127.0.0.1"),
		9000,
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	// session.mu.RLock()
	// localPort := session.conn.LocalAddr().(*net.UDPAddr).Port
	// session.mu.RUnlock()

	receiver, err := net.ListenUDP(
		"udp",
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: 9000,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer receiver.Close()

	const writers = 20

	var wg sync.WaitGroup
	wg.Add(writers)

	for i := 0; i < writers; i++ {
		go func() {
			defer wg.Done()

			err := session.Write(
				[]byte{0x01, 0x02, 0x03},
				160,
			)

			if err != nil {
				t.Errorf("Write failed: %v", err)
			}
		}()
	}

	wg.Wait()

	// Read all packets.
	receiver.SetReadDeadline(
		time.Now().Add(500 * time.Millisecond),
	)

	sequences := make(map[uint16]bool)

	for i := 0; i < writers; i++ {
		buf := make([]byte, 2048)

		n, _, err := receiver.ReadFromUDP(buf)
		if err != nil {
			t.Fatalf("failed to receive packet %d: %v", i, err)
		}

		packet := &RTPPacket{}

		if err := packet.Unmarshal(buf[:n]); err != nil {
			t.Fatalf("invalid RTP packet: %v", err)
		}

		seq := packet.Header.SequenceNumber

		if sequences[seq] {
			t.Fatalf("duplicate sequence number: %d", seq)
		}

		sequences[seq] = true
	}

	if len(sequences) != writers {
		t.Fatalf(
			"expected %d unique sequences, got %d",
			writers,
			len(sequences),
		)
	}
}
func TestIntegrationSIPRTPSTT(t *testing.T) {

	stt := &integrationSTT{
		called: make(chan neurocall.AudioSegment, 1),
	}

	config := SIPConfig{
		ListenIP:   "127.0.0.1",
		SIPPort:    0,
		RTPMinPort: 20000,
		RTPMaxPort: 20100,
	}

	server := NewSIPServer(
		config,
		nil,
		nil,
		stt,
		nil,
		nil,
	)

	ctx, cancel := context.WithCancel(
		context.Background(),
	)
	defer cancel()

	if err := server.Listen(ctx); err != nil {
		t.Fatal(err)
	}

	defer server.Stop()

	server.mu.RLock()
	sipAddr := server.conn.LocalAddr().(*net.UDPAddr)
	server.mu.RUnlock()

	t.Logf(
		"SIP server listening on %s",
		sipAddr,
	)

	if sipAddr.Port == 0 {
		t.Fatal("SIP server did not receive a port")
	}

	// ادامه مرحله بعد...
}
func TestIntegrationSIPInvite200OK(t *testing.T) {
	stt := &integrationSTT{
		called: make(chan neurocall.AudioSegment, 1),
	}

	config := SIPConfig{
		ListenIP:   "127.0.0.1",
		SIPPort:    0,
		RTPMinPort: 21000,
		RTPMaxPort: 21100,
	}

	server := NewSIPServer(
		config,
		nil,
		nil,
		stt,
		nil,
		nil,
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := server.Listen(ctx); err != nil {
		t.Fatal(err)
	}
	defer server.Stop()

	server.mu.RLock()
	sipAddr := server.conn.LocalAddr().(*net.UDPAddr)
	server.mu.RUnlock()

	client, err := net.ListenUDP(
		"udp",
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: 0,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	callID := "integration-invite-200"

	invite := []byte(
		"INVITE sip:neurocall@127.0.0.1 SIP/2.0\r\n" +
			"Via: SIP/2.0/UDP 127.0.0.1:" +
			strconv.Itoa(client.LocalAddr().(*net.UDPAddr).Port) +
			";branch=z9hG4bK-integration\r\n" +
			"From: <sip:test@127.0.0.1>;tag=test\r\n" +
			"To: <sip:neurocall@127.0.0.1>\r\n" +
			"Call-ID: " + callID + "\r\n" +
			"CSeq: 1 INVITE\r\n" +
			"Contact: <sip:test@127.0.0.1>\r\n" +
			"Content-Type: application/sdp\r\n" +
			"Content-Length: 0\r\n" +
			"\r\n",
	)

	// Replace empty SDP with a valid SDP body.
	body := []byte(
		"v=0\r\n" +
			"o=test 1 1 IN IP4 127.0.0.1\r\n" +
			"s=test\r\n" +
			"c=IN IP4 127.0.0.1\r\n" +
			"t=0 0\r\n" +
			"m=audio 22000 RTP/AVP 0 8\r\n" +
			"a=rtpmap:0 PCMU/8000/1\r\n" +
			"a=rtpmap:8 PCMA/8000/1\r\n",
	)

	invite = []byte(
		strings.Replace(
			string(invite),
			"Content-Length: 0\r\n\r\n",
			"Content-Length: "+strconv.Itoa(len(body))+"\r\n\r\n"+
				string(body),
			1,
		),
	)

	if _, err := client.WriteToUDP(invite, sipAddr); err != nil {
		t.Fatal(err)
	}

	client.SetReadDeadline(time.Now().Add(2 * time.Second))

	buf := make([]byte, 65535)

	n, _, err := client.ReadFromUDP(buf)
	if err != nil {
		t.Fatal(err)
	}

	response := string(buf[:n])

	if !strings.HasPrefix(response, "SIP/2.0 100 Trying") {
		t.Fatalf(
			"expected 100 Trying, got:\n%s",
			response,
		)
	}

	n, _, err = client.ReadFromUDP(buf)
	if err != nil {
		t.Fatal(err)
	}

	response = string(buf[:n])

	if !strings.HasPrefix(response, "SIP/2.0 200 OK") {
		t.Fatalf(
			"expected 200 OK, got:\n%s",
			response,
		)
	}

	call, err := server.manager.Get(callID)
	if err != nil {
		t.Fatal(err)
	}

	if call == nil {
		t.Fatal("expected call")
	}

	if call.Codec.Name != "PCMU" {
		t.Fatalf(
			"expected PCMU codec, got %s",
			call.Codec.Name,
		)
	}

	if call.LocalRTPPort == 0 {
		t.Fatal("expected allocated RTP port")
	}
}
func TestIntegrationSIPInviteACK(t *testing.T) {
	stt := &integrationSTT{
		called: make(chan neurocall.AudioSegment, 1),
	}

	config := SIPConfig{
		ListenIP:   "127.0.0.1",
		SIPPort:    0,
		RTPMinPort: 21200,
		RTPMaxPort: 21300,
	}

	server := NewSIPServer(
		config,
		nil,
		nil,
		stt,
		nil,
		nil,
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := server.Listen(ctx); err != nil {
		t.Fatal(err)
	}
	defer server.Stop()

	server.mu.RLock()
	sipAddr := server.conn.LocalAddr().(*net.UDPAddr)
	server.mu.RUnlock()

	client, err := net.ListenUDP(
		"udp",
		&net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: 0,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	callID := "integration-ack"

	sdpBody := []byte(
		"v=0\r\n" +
			"o=test 1 1 IN IP4 127.0.0.1\r\n" +
			"s=test\r\n" +
			"c=IN IP4 127.0.0.1\r\n" +
			"t=0 0\r\n" +
			"m=audio 22001 RTP/AVP 0\r\n" +
			"a=rtpmap:0 PCMU/8000/1\r\n",
	)

	invite := buildIntegrationINVITE(
		callID,
		client.LocalAddr().(*net.UDPAddr).Port,
		sdpBody,
	)

	if _, err := client.WriteToUDP(invite, sipAddr); err != nil {
		t.Fatal(err)
	}

	client.SetReadDeadline(time.Now().Add(2 * time.Second))

	buf := make([]byte, 65535)

	// 100 Trying
	if _, _, err := client.ReadFromUDP(buf); err != nil {
		t.Fatal(err)
	}

	// 200 OK
	n, _, err := client.ReadFromUDP(buf)
	if err != nil {
		t.Fatal(err)
	}

	response := string(buf[:n])

	if !strings.HasPrefix(response, "SIP/2.0 200 OK") {
		t.Fatalf(
			"expected 200 OK, got:\n%s",
			response,
		)
	}

	ack := []byte(
		"ACK sip:neurocall@127.0.0.1 SIP/2.0\r\n" +
			"Via: SIP/2.0/UDP 127.0.0.1:" +
			strconv.Itoa(client.LocalAddr().(*net.UDPAddr).Port) +
			";branch=z9hG4bK-ack\r\n" +
			"From: <sip:test@127.0.0.1>;tag=test\r\n" +
			"To: <sip:neurocall@127.0.0.1>\r\n" +
			"Call-ID: " + callID + "\r\n" +
			"CSeq: 1 ACK\r\n" +
			"Content-Length: 0\r\n" +
			"\r\n",
	)

	if _, err := client.WriteToUDP(ack, sipAddr); err != nil {
		t.Fatal(err)
	}

	time.Sleep(50 * time.Millisecond)

	call, err := server.manager.Get(callID)
	if err != nil {
		t.Fatal(err)
	}

	if !call.Answered() {
		t.Fatal("expected call to be answered after ACK")
	}

	session, err := server.manager.GetSession(callID)
	if err != nil {
		t.Fatal(err)
	}

	if !session.Running() {
		t.Fatal("expected session to be running")
	}
}
func TestIntegrationSIPInviteCreatesCall(t *testing.T) {
	manager := NewCallManager(20000, 20100)
	codecs := NewCodecRegistry()

	server := NewSIPServer(
		SIPConfig{
			ListenIP:   "127.0.0.1",
			SIPPort:    0,
			RTPMinPort: 20000,
			RTPMaxPort: 20100,
			ExternalIP: "127.0.0.1",
		},
		manager,
		codecs,
		&FakeSTT{},
		nil,
		nil,
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := server.Listen(ctx); err != nil {
		t.Fatal(err)
	}
	defer server.Stop()

	server.mu.RLock()
	conn := server.conn
	server.mu.RUnlock()

	if conn == nil {
		t.Fatal("SIP server connection is nil")
	}

	sipAddr := conn.LocalAddr().(*net.UDPAddr)

	client, err := net.DialUDP(
		"udp",
		nil,
		sipAddr,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	invite := []byte(
		"INVITE sip:test@127.0.0.1 SIP/2.0\r\n" +
			"Via: SIP/2.0/UDP 127.0.0.1:6000\r\n" +
			"From: <sip:test@127.0.0.1>\r\n" +
			"To: <sip:test@127.0.0.1>\r\n" +
			"Call-ID: integration-test-call\r\n" +
			"CSeq: 1 INVITE\r\n" +
			"Content-Type: application/sdp\r\n" +
			"Content-Length: 96\r\n" +
			"\r\n" +
			"v=0\r\n" +
			"o=test 1 1 IN IP4 127.0.0.1\r\n" +
			"s=test\r\n" +
			"c=IN IP4 127.0.0.1\r\n" +
			"t=0 0\r\n" +
			"m=audio 9000 RTP/AVP 0\r\n" +
			"a=rtpmap:0 PCMU/8000/1\r\n",
	)

	if _, err := client.Write(invite); err != nil {
		t.Fatal(err)
	}

	client.SetReadDeadline(
		time.Now().Add(2 * time.Second),
	)

	buffer := make([]byte, 65535)

	var response string

	for {
		n, _, err := client.ReadFromUDP(buffer)
		if err != nil {
			t.Fatal(err)
		}

		response = string(buffer[:n])

		if strings.HasPrefix(response, "SIP/2.0 200") {
			break
		}

		if strings.HasPrefix(response, "SIP/2.0 503") {
			t.Fatalf("server returned 503:\n%s", response)
		}

		if strings.HasPrefix(response, "SIP/2.0 488") {
			t.Fatalf("server returned 488:\n%s", response)
		}
	}

	if !strings.Contains(response, "m=audio") {
		t.Fatalf("200 OK does not contain SDP:\n%s", response)
	}

	call, err := manager.Get("integration-test-call")
	if err != nil {
		t.Fatalf("call was not created: %v", err)
	}

	if call.LocalRTPPort == 0 {
		t.Fatal("expected allocated RTP port")
	}

	session, err := manager.GetSession("integration-test-call")
	if err != nil {
		t.Fatalf("session was not created: %v", err)
	}

	if session == nil {
		t.Fatal("session is nil")
	}

	if !session.Running() {
		t.Fatal("expected session to be running")
	}

	if call.Codec.Name != "PCMU" {
		t.Fatalf(
			"expected PCMU codec, got %s",
			call.Codec.Name,
		)
	}

	_ = neurocall.Codec{}
}
func TestIntegrationSIPACKRTPToSTT(t *testing.T) {
	manager := NewCallManager(21000, 21100)
	codecs := NewCodecRegistry()
	stt := &testSTT{
		transcribe: func(
			ctx context.Context,
			segment neurocall.AudioSegment,
		) (neurocall.Transcript, error) {

			if len(segment.Data) == 0 {
				t.Error("STT received empty segment")
			}

			return neurocall.Transcript{
				Text: "integration speech",
			}, nil
		},
	}

	server := NewSIPServer(
		SIPConfig{
			ListenIP:   "127.0.0.1",
			SIPPort:    0,
			RTPMinPort: 21000,
			RTPMaxPort: 21100,
			ExternalIP: "127.0.0.1",
		},
		manager,
		codecs,
		stt,
		nil,
		nil,
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := server.Listen(ctx); err != nil {
		t.Fatal(err)
	}
	defer server.Stop()

	server.mu.RLock()
	sipConn := server.conn
	server.mu.RUnlock()

	if sipConn == nil {
		t.Fatal("SIP connection is nil")
	}

	sipAddr := sipConn.LocalAddr().(*net.UDPAddr)

	client, err := net.DialUDP(
		"udp",
		nil,
		sipAddr,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	callID := "integration-ack-rtp-stt"

	sdpBody :=
		"v=0\r\n" +
			"o=test 1 1 IN IP4 127.0.0.1\r\n" +
			"s=test\r\n" +
			"c=IN IP4 127.0.0.1\r\n" +
			"t=0 0\r\n" +
			"m=audio 9000 RTP/AVP 0\r\n" +
			"a=rtpmap:0 PCMU/8000/1\r\n"

	invite :=
		"INVITE sip:test@127.0.0.1 SIP/2.0\r\n" +
			"Via: SIP/2.0/UDP 127.0.0.1:6000\r\n" +
			"From: <sip:test@127.0.0.1>\r\n" +
			"To: <sip:test@127.0.0.1>\r\n" +
			"Call-ID: " + callID + "\r\n" +
			"CSeq: 1 INVITE\r\n" +
			"Content-Type: application/sdp\r\n" +
			"Content-Length: " +
			strconv.Itoa(len(sdpBody)) +
			"\r\n\r\n" +
			sdpBody

	if _, err := client.Write([]byte(invite)); err != nil {
		t.Fatal(err)
	}

	client.SetReadDeadline(
		time.Now().Add(2 * time.Second),
	)

	buffer := make([]byte, 65535)

	var response string

	for {
		n, _, err := client.ReadFromUDP(buffer)
		if err != nil {
			t.Fatal(err)
		}

		response = string(buffer[:n])

		if strings.HasPrefix(response, "SIP/2.0 200") {
			break
		}

		if strings.HasPrefix(response, "SIP/2.0 503") {
			t.Fatalf("received 503:\n%s", response)
		}

		if strings.HasPrefix(response, "SIP/2.0 488") {
			t.Fatalf("received 488:\n%s", response)
		}
	}

	call, err := manager.Get(callID)
	if err != nil {
		t.Fatal(err)
	}

	if call.LocalRTPPort == 0 {
		t.Fatal("RTP port was not allocated")
	}

	session, err := manager.GetSession(callID)
	if err != nil {
		t.Fatal(err)
	}

	transcriptCh := make(chan TranscriptEvent, 1)

	session.Events.Subscribe(
		EventTranscript,
		func(event Event) {
			data, ok := event.Data.(TranscriptEvent)
			if !ok {
				return
			}

			select {
			case transcriptCh <- data:
			default:
			}
		},
	)

	ack :=
		"ACK sip:test@127.0.0.1 SIP/2.0\r\n" +
			"Via: SIP/2.0/UDP 127.0.0.1:6000\r\n" +
			"From: <sip:test@127.0.0.1>\r\n" +
			"To: <sip:test@127.0.0.1>\r\n" +
			"Call-ID: " + callID + "\r\n" +
			"CSeq: 1 ACK\r\n" +
			"Content-Length: 0\r\n\r\n"

	if _, err := client.Write([]byte(ack)); err != nil {
		t.Fatal(err)
	}

	time.Sleep(50 * time.Millisecond)

	if !call.Answered() {
		t.Fatal("call was not answered after ACK")
	}

	/*
		Use the RTP port selected by the SIP server.
	*/
	rtpAddr := &net.UDPAddr{
		IP:   net.ParseIP("127.0.0.1"),
		Port: call.LocalRTPPort,
	}

	rtpClient, err := net.DialUDP(
		"udp",
		nil,
		rtpAddr,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rtpClient.Close()

	/*
		PCMU payload.

		0x00 is decoded as a strong negative PCM value,
		which is enough to trigger the energy VAD.
	*/
	// Send speech frames.
	speechPayload := make([]byte, 160)

	for i := range speechPayload {
		speechPayload[i] = 0x00
	}

	for seq := uint16(1); seq <= 20; seq++ {

		packet := &RTPPacket{
			Header: RTPHeader{
				Version:        2,
				PayloadType:    0,
				SequenceNumber: seq,
				Timestamp:      uint32(seq-1) * 160,
				SSRC:           12345,
			},
			Payload: speechPayload,
		}

		data, err := packet.Marshal()
		if err != nil {
			t.Fatal(err)
		}

		if _, err := rtpClient.Write(data); err != nil {
			t.Fatal(err)
		}

		time.Sleep(5 * time.Millisecond)
	}

	// Send silence frames.
	// PCMU 0xFF decodes close to PCM zero.
	silencePayload := make([]byte, 160)

	for i := range silencePayload {
		silencePayload[i] = 0xFF
	}

	for seq := uint16(21); seq <= 30; seq++ {

		packet := &RTPPacket{
			Header: RTPHeader{
				Version:        2,
				PayloadType:    0,
				SequenceNumber: seq,
				Timestamp:      uint32(seq-1) * 160,
				SSRC:           12345,
			},
			Payload: silencePayload,
		}

		data, err := packet.Marshal()
		if err != nil {
			t.Fatal(err)
		}

		if _, err := rtpClient.Write(data); err != nil {
			t.Fatal(err)
		}

		time.Sleep(5 * time.Millisecond)
	}

	select {
	case event := <-transcriptCh:

		if event.CallID != callID {
			t.Fatalf(
				"expected CallID %q, got %q",
				callID,
				event.CallID,
			)
		}

		if event.Transcript.Text != "integration speech" {
			t.Fatalf(
				"expected transcript %q, got %q",
				"integration speech",
				event.Transcript.Text,
			)
		}

	case <-time.After(3 * time.Second):
		t.Fatal(
			"timed out waiting for TranscriptEvent",
		)
	}
}
