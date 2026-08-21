package core

import (
	"bytes"
	"net"
	"strings"
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

	// RTP version = 1
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
	// Version 2 + extension bit
	data[0] = 0x90
	// Extension length = 2 words = 8 bytes
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
	engine := NewSTTEngine(
		NewFakeSTT(),
	)
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
func TestRTPSequenceTrackerLoss(t *testing.T) {
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

	// Arrive out of order.
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

	// Duplicate must not create another packet.
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
