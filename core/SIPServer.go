package core

import (
	"context"
	"fmt"
	"net"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewSIPServer(
	config SIPConfig,
	manager *CallManager,
	codecs *CodecRegistry,
	stt neurocall.STT,
	llm neurocall.LLM,
	tts neurocall.TTS,
) *SIPServer {

	if manager == nil {
		manager = NewCallManager(
			config.RTPMinPort,
			config.RTPMaxPort,
		)
	}

	if codecs == nil {
		codecs = NewCodecRegistry()
	}
	bus := NewEventBus()
	server := &SIPServer{
		manager: manager,
		codecs:  codecs,
		config:  config,
		bus:     bus,
	}
	manager.SetEventBus(bus)
	if stt != nil {
		server.stt = NewSTTEngine(stt)
	}

	if llm != nil {
		server.llm = NewLLMEngine(llm)
	}

	if tts != nil {
		server.tts = NewTTSEngine(tts)
	}
	if server.llm != nil {
		server.conversation =
			NewConversationEngine(
				server.llm,
				bus,
			)
	}

	if server.tts != nil {
		server.voice =
			NewVoiceResponseEngine(
				server.tts,
				bus,
			)
	}

	return server
}
func (s *SIPServer) Listen(ctx context.Context) error {

	s.mu.Lock()

	if s.conn != nil {
		s.mu.Unlock()

		return fmt.Errorf(
			"SIP server already listening",
		)
	}

	addr := &net.UDPAddr{
		IP: net.ParseIP(
			s.config.ListenIP,
		),
		Port: s.config.SIPPort,
	}

	conn, err := net.ListenUDP(
		"udp",
		addr,
	)

	if err != nil {
		s.mu.Unlock()
		return err
	}

	serverCtx, cancel :=
		context.WithCancel(ctx)

	s.conn = conn
	s.ctx = serverCtx
	s.cancel = cancel

	s.mu.Unlock()

	go s.readLoop(serverCtx)

	return nil
}
func (s *SIPServer) Stop() error {
	s.mu.Lock()
	conn := s.conn
	cancel := s.cancel
	s.conn = nil
	s.cancel = nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if conn != nil {
		_ = conn.Close()
	}
	if s.manager != nil {
		for _, call := range s.manager.List() {
			if call == nil {
				continue
			}
			_ = s.manager.Remove(call.CallID)
		}
	}
	return nil
}
func (s *SIPServer) readLoop(ctx context.Context) {

	buffer := make(
		[]byte,
		65535,
	)

	for {

		select {
		case <-ctx.Done():
			return

		default:
		}

		s.mu.RLock()
		conn := s.conn
		s.mu.RUnlock()

		if conn == nil {
			return
		}

		n, remote, err :=
			conn.ReadFromUDP(buffer)

		if err != nil {
			select {
			case <-ctx.Done():
				return
			default:
				continue
			}
		}

		data := append(
			[]byte(nil),
			buffer[:n]...,
		)

		go func(
			data []byte,
			remote *net.UDPAddr,
		) {
			if err := s.handle(ctx, data, remote); err != nil {
				fmt.Printf(
					"SIP handle error from %s: %v\n",
					remote,
					err,
				)
			}
		}(data, remote)
	}
}
func (s *SIPServer) handle(ctx context.Context, data []byte, remote *net.UDPAddr) error {
	message, err := ParseSIPMessage(data)
	if err != nil {
		return err
	}
	switch message.Method() {
	case "INVITE":
		return s.handleINVITE(
			ctx,
			message,
			remote,
		)
	case "ACK":
		return s.handleACK(
			ctx,
			message,
			remote,
		)
	case "BYE":
		return s.handleBYE(
			ctx,
			message,
			remote,
		)
	case "OPTIONS":
		return s.handleOPTIONS(
			ctx,
			message,
			remote,
		)
	default:
		return s.sendResponse(
			message,
			remote,
			501,
			"Not Implemented",
			nil,
		)
	}
}
func (s *SIPServer) sendResponse(req *SIPMessage, remote *net.UDPAddr, code int, reason string, body []byte) error {

	data := BuildSIPResponse(
		req,
		code,
		reason,
		body,
	)

	s.mu.RLock()
	conn := s.conn
	s.mu.RUnlock()

	if conn == nil {
		return neurocall.ErrCallClosed
	}

	_, err := conn.WriteToUDP(
		data,
		remote,
	)

	return err
}
func (s *SIPServer) handleINVITE(_ context.Context, req *SIPMessage, remote *net.UDPAddr) error {
	if err := s.sendResponse(req, remote, 100, "Trying", nil); err != nil {
		return err
	}
	if s.stt == nil {
		return s.sendResponse(req, remote, 503, "Service Unavailable", nil)
	}
	media, err := ParseSDP(req.Body)
	if err != nil {
		return s.sendResponse(req, remote, 488, "Not Acceptable Here", nil)
	}
	codec, err := s.codecs.Negotiate(media.Codecs)
	if err != nil {
		return s.sendResponse(req, remote, 488, "Not Acceptable Here", nil)
	}
	callID := req.Headers["call-id"]
	if callID == "" {
		return s.sendResponse(req, remote, 400, "Bad Request", nil)
	}
	var ttsPlayback *TTSPlayback
	if s.tts != nil {
		ttsPlayback, err = NewTTSPlayback(s.tts)
		if err != nil {
			return err
		}
	}
	cleanup := func() {
		if s.voice != nil {
			_ = s.voice.RemoveCall(callID)
		}

		_ = s.manager.Remove(callID)
	}
	// --------------------
	// Create call
	// --------------------
	call := &SIPCall{
		CallID:     callID,
		RemoteIP:   media.IP,
		RemotePort: media.Port,
		RemoteAddr: remote,
		Codec:      codec,
		TTS:        ttsPlayback,
	}
	if s.voice != nil {
		if err := s.voice.AddCall(call); err != nil {
			cleanup()
			return err
		}
	}
	// --------------------
	// Register call
	// --------------------
	if err := s.manager.Add(call); err != nil {
		cleanup()
		return err
	}
	session, err := s.manager.CreateSession(call)
	if err != nil {
		cleanup()
		return err
	}

	sttWorker, err := NewSTTWorker(
		s.stt,
		session.Events,
		callID,
		16,
	)
	if err != nil {
		cleanup()
		return err
	}

	if err := session.SetSTTWorker(sttWorker); err != nil {
		cleanup()
		return err
	}

	if err := call.SetSTTWorker(sttWorker); err != nil {
		cleanup()
		return err
	}

	if s.llm != nil {
		if err := session.ConfigureConversation(s.llm); err != nil {
			cleanup()
			return err
		}
	}

	if s.tts != nil {
		if err := session.ConfigureVoice(s.tts); err != nil {
			cleanup()
			return err
		}
	}

	// Allocate RTP
	port, err := s.manager.ports.Allocate()
	if err != nil {
		cleanup()
		return err
	}

	call.mu.Lock()
	call.LocalRTPPort = port
	call.mu.Unlock()

	// Start RTP
	if err := call.StartRTP(s.config.ListenIP); err != nil {
		cleanup()
		return err
	}

	// Start Audio
	if err := call.StartAudio(s.configToAudioConfig()); err != nil {
		cleanup()
		return err
	}

	// VAD + Segmenter
	vad := NewEnergyVAD(500)

	segmenter := NewAudioSegmenter(
		vad,
		s.configToAudioConfig(),
	)

	call.mu.Lock()
	call.VAD = vad
	call.Segmenter = segmenter
	call.mu.Unlock()

	// Start Session
	if err := session.Start(); err != nil {
		cleanup()
		return err
	}

	// Start RTP receive → STT
	// call.StartReceiveLoop()

	// SDP Answer
	body := BuildSDPAnswer(call, s.config)
	// --------------------
	// 200 OK
	// --------------------
	if err := s.sendResponse(req, remote, 200, "OK", body); err != nil {
		cleanup()
		return err
	}
	return nil
}
func (s *SIPServer) handleACK(_ context.Context, req *SIPMessage, _ *net.UDPAddr) error {
	callID := req.Headers["call-id"]
	if callID == "" {
		return neurocall.ErrCallNotFound
	}
	call, err := s.manager.Get(callID)
	if err != nil {
		return err
	}
	if !call.Answered() {
		if err := call.Answer(); err != nil {
			return err
		}
		call.StartReceiveLoop()
	}
	session, err := s.manager.GetSession(callID)
	if err == nil {
		session.Events.Publish(Event{
			Name: EventCallAnswered,
			Data: session,
		})
	}
	return nil
}
func (s *SIPServer) handleBYE(_ context.Context, req *SIPMessage, remote *net.UDPAddr) error {
	callID := req.Headers["call-id"]
	if callID == "" {
		return s.sendResponse(req, remote, 400, "Bad Request", nil)
	}
	call, err := s.manager.Get(callID)
	if err != nil {
		return s.sendResponse(req, remote, 481, "Call/Transaction Does Not Exist", nil)
	}
	if err := s.sendResponse(req, remote, 200, "OK", nil); err != nil {
		return err
	}
	_ = call.Hangup()
	if s.voice != nil {
		_ = s.voice.RemoveCall(callID)
	}
	return s.manager.Remove(callID)
}
func (s *SIPServer) handleOPTIONS(_ context.Context, req *SIPMessage, remote *net.UDPAddr) error {

	return s.sendResponse(
		req,
		remote,
		200,
		"OK",
		nil,
	)
}
func (s *SIPServer) MediaIP() string {
	if s.config.ExternalIP != "" {
		return s.config.ExternalIP
	}
	if s.config.ListenIP != "" &&
		s.config.ListenIP != "0.0.0.0" &&
		s.config.ListenIP != "::" {
		return s.config.ListenIP
	}
	return "127.0.0.1"
}
func (s *SIPServer) configToAudioConfig() AudioConfig {
	return AudioConfig{
		SampleRate: DefaultAudioSampleRate,
		Channels:   DefaultAudioChannels,
		FrameSize:  DefaultAudioFrameSize,
		BufferSize: 32,
	}
}
