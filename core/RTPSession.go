package core

import (
	"context"
	"fmt"
	"net"
	"time"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewRTPSession(
	localIP string,
	port int,
	remoteIP net.IP,
	remotePort int,
	codec neurocall.Codec,
) (*RTPSession, error) {
	if !codec.Valid() {
		return nil, fmt.Errorf(
			"invalid codec",
		)
	}
	if remoteIP == nil {
		return nil, fmt.Errorf("invalid remote IP: %s", remoteIP)
	}
	localAddr := &net.UDPAddr{
		IP:   net.ParseIP(localIP),
		Port: port,
	}
	if localAddr.IP == nil {
		return nil, fmt.Errorf("invalid local IP: %s", localIP)
	}
	conn, err := net.ListenUDP("udp", localAddr)
	if err != nil {
		return nil, err
	}
	remote := &net.UDPAddr{
		IP:   remoteIP,
		Port: remotePort,
	}
	return &RTPSession{
		conn:            conn,
		remote:          remote,
		codec:           codec,
		ssrc:            randomUint32(),
		sequence:        0,
		timestamp:       0,
		sequenceTracker: &RTPSequenceTracker{},
		jitter:          NewRTPJitterBuffer(50, 30*time.Millisecond),
		writeInterval:   20 * time.Millisecond,
	}, nil
}
func (s *RTPSession) ReadOrdered() (RTPReadResult, error) {
	for {
		packet, ready := s.jitter.Pop()
		if ready {
			if packet == nil {
				return RTPReadResult{
					Lost: true,
				}, nil
			}
			return RTPReadResult{
				Packet: packet,
			}, nil
		}
		if s.jitter.HasPackets() {
			s.jitter.WaitForChange(s.jitter.maxWait)
			continue
		}
		packet, err := s.Read()
		if err != nil {
			return RTPReadResult{}, err
		}
		if err := s.jitter.Push(packet); err != nil {
			continue
		}
	}
}
func (s *RTPSession) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return nil
	}
	return conn.Close()
}
func (s *RTPSession) ReadPCM(decoder neurocall.Decoder) ([]int16, error) {

	if decoder == nil {
		return nil, fmt.Errorf("decoder is nil")
	}

	s.mu.RLock()

	if s.closed {
		s.mu.RUnlock()
		return nil, errRTPSessionClosed
	}

	codec := s.codec

	s.mu.RUnlock()

	result, err := s.ReadOrdered()

	if err != nil {
		return nil, err
	}

	if result.Lost {

		frameSamples := codec.ClockRate / 50

		if frameSamples <= 0 {
			frameSamples = 160
		}

		return concealPCM(
			codec,
			frameSamples,
		), nil
	}

	packet := result.Packet

	if packet.Header.PayloadType != codec.PayloadType {
		return nil, fmt.Errorf(
			"unexpected RTP payload type: %d",
			packet.Header.PayloadType,
		)
	}

	pcm := decoder.Decode(packet.Payload)

	if len(pcm) == 0 {
		return nil, neurocall.ErrInvalidAudio
	}

	return pcm, nil
}
func (s *RTPSession) WritePCM(encoder neurocall.Encoder, pcm []int16) error {
	if encoder == nil {
		return fmt.Errorf("encoder is nil")
	}
	if len(pcm) == 0 {
		return neurocall.ErrInvalidAudio
	}
	payload := encoder.Encode(pcm)
	if len(payload) == 0 {
		return neurocall.ErrInvalidAudio
	}
	return s.Write(payload, len(pcm))
}
func (s *RTPSession) AdvanceTimestamp(samples int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.timestamp += uint32(samples)
}
func (s *RTPSession) Stats() RTPStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.stats
}
func (s *RTPSession) ResetStats() {
	s.mu.Lock()
	s.stats = RTPStats{}
	tracker := s.sequenceTracker
	s.mu.Unlock()
	if tracker != nil {
		tracker.Reset()
	}
}
func (s *RTPSession) Read() (*RTPPacket, error) {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return nil, errRTPSessionClosed
	}
	conn := s.conn
	s.mu.RUnlock()
	if conn == nil {
		return nil, errRTPSessionClosed
	}
	buf := make([]byte, 2048)
	n, _, err := conn.ReadFromUDP(buf)
	if err != nil {
		s.mu.RLock()
		closed := s.closed
		s.mu.RUnlock()
		if closed {
			return nil, errRTPSessionClosed
		}
		return nil, err
	}
	s.mu.Lock()
	s.stats.BytesReceived += uint64(n)
	s.mu.Unlock()
	packet := &RTPPacket{}
	if err := packet.Unmarshal(buf[:n]); err != nil {
		s.mu.Lock()
		s.stats.InvalidPackets++
		s.mu.Unlock()
		return nil, err
	}
	s.sequenceTracker.Update(packet.Header.SequenceNumber)
	trackerStats := s.sequenceTracker.Stats()
	s.mu.Lock()
	s.stats.ReceivedPackets++
	s.stats.LostPackets = trackerStats.LostPackets
	s.mu.Unlock()
	return packet, nil
}
func (s *RTPSession) Write(payload []byte, samples int) error {

	if len(payload) == 0 {
		return neurocall.ErrInvalidAudio
	}

	if samples <= 0 {
		return neurocall.ErrInvalidAudio
	}

	s.mu.Lock()

	if s.closed {
		s.mu.Unlock()
		return errRTPSessionClosed
	}

	conn := s.conn
	remote := s.remote

	packet := RTPPacket{
		Header: RTPHeader{
			Version:        2,
			PayloadType:    s.codec.PayloadType,
			SequenceNumber: s.sequence,
			Timestamp:      s.timestamp,
			SSRC:           s.ssrc,
		},
		Payload: append([]byte(nil), payload...),
	}

	data, err := packet.Marshal()

	if err != nil {
		s.mu.Unlock()
		return err
	}

	s.sequence++
	s.timestamp += uint32(samples)

	s.mu.Unlock()

	if _, err := conn.WriteToUDP(
		data,
		remote,
	); err != nil {
		return err
	}

	s.mu.Lock()

	s.stats.SentPackets++
	s.stats.BytesSent += uint64(len(data))

	s.mu.Unlock()

	return nil
}
func (s *RTPSession) WritePCMRealtime(ctx context.Context, encoder neurocall.Encoder, pcm []int16) error {

	if encoder == nil {
		return fmt.Errorf("encoder is nil")
	}

	if len(pcm) == 0 {
		return neurocall.ErrInvalidAudio
	}
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.RLock()

	interval := s.writeInterval

	s.mu.RUnlock()

	if interval <= 0 {
		interval = 20 * time.Millisecond
	}

	sampleRate := s.codec.ClockRate

	if sampleRate <= 0 {
		sampleRate = 8000
	}

	frameSize := sampleRate / 50

	if frameSize <= 0 {
		frameSize = 160
	}

	ticker := time.NewTicker(interval)

	defer ticker.Stop()

	first := true

	for offset := 0; offset < len(pcm); offset += frameSize {

		end := offset + frameSize

		if end > len(pcm) {
			end = len(pcm)
		}

		frame := pcm[offset:end]

		if !first {
			select {
			case <-ctx.Done():
				return ctx.Err()

			case <-ticker.C:
			}
		}

		first = false

		if err := s.WritePCM(
			encoder,
			frame,
		); err != nil {
			return err
		}
	}

	return nil
}
