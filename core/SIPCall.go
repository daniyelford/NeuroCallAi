package core

import (
	"context"
	"fmt"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func (c *SIPCall) ID() string {
	return c.CallID
}
func (c *SIPCall) Answer() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return neurocall.ErrCallClosed
	}
	if c.answered {
		return nil
	}
	c.answered = true
	return nil
}
func (c *SIPCall) Hangup() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	rtp := c.RTP
	pipeline := c.Pipeline
	stt := c.STT
	c.mu.Unlock()
	if stt != nil {
		stt.Stop()
	}
	if rtp != nil {
		_ = rtp.Close()
	}
	if pipeline != nil {
		_ = pipeline.Close()
	}
	return nil
}
func (c *SIPCall) Answered() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.answered && !c.closed
}
func (c *SIPCall) Closed() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.closed
}
func (c *SIPCall) SetSTTWorker(
	worker *STTWorker,
) error {

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return neurocall.ErrCallClosed
	}

	c.STT = worker

	return nil
}
func (c *SIPCall) STTWorker() *STTWorker {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.STT
}
func (c *SIPCall) StartReceiveLoop() {
	if c == nil {
		return
	}
	c.mu.Lock()
	if c.closed || c.receiveRunning {
		c.mu.Unlock()
		return
	}
	c.receiveRunning = true
	c.mu.Unlock()
	go func() {
		defer func() {
			c.mu.Lock()
			c.receiveRunning = false
			c.mu.Unlock()
		}()
		for {
			c.mu.RLock()
			if c.closed {
				c.mu.RUnlock()
				return
			}
			rtp := c.RTP
			pipeline := c.Pipeline
			segmenter := c.Segmenter
			stt := c.STT
			codec := c.Codec
			c.mu.RUnlock()
			if rtp == nil ||
				pipeline == nil ||
				segmenter == nil {
				return
			}
			decoder := pipeline.Decoder()
			if decoder == nil {
				return
			}
			pcm, err := rtp.ReadPCM(decoder)
			if err != nil {
				if c.Closed() {
					return
				}
				continue
			}
			if len(pcm) == 0 {
				continue
			}
			frame := neurocall.AudioFrame{
				Data:       pcm,
				SampleRate: codec.ClockRate,
				Channels:   codec.Channels,
			}
			ctx := context.Background()
			segments, err := segmenter.Process(ctx, frame)
			if err != nil {
				continue
			}
			if stt == nil {
				continue
			}
			for _, segment := range segments {
				if len(segment.Data) == 0 {
					continue
				}
				if err := stt.Push(segment); err != nil {
					if c.Closed() {
						return
					}

					continue
				}
			}
		}
	}()
}
func (c *SIPCall) StartRTP(
	localIP string,
) error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return neurocall.ErrCallClosed
	}
	if c.RTP != nil {
		c.mu.Unlock()
		return nil
	}
	codec := c.Codec
	remoteIP := c.RemoteIP
	remotePort := c.RemotePort
	localPort := c.LocalRTPPort
	c.mu.Unlock()
	rtp, err := NewRTPSession(
		localIP,
		localPort,
		remoteIP,
		remotePort,
		codec,
	)
	if err != nil {
		return err
	}
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		_ = rtp.Close()
		return neurocall.ErrCallClosed
	}
	c.RTP = rtp
	c.mu.Unlock()
	return nil
}
func (c *SIPCall) StartAudio(
	cfg AudioConfig,
) error {

	c.mu.Lock()

	if c.closed {
		c.mu.Unlock()
		return neurocall.ErrCallClosed
	}

	if c.Pipeline != nil {
		c.mu.Unlock()
		return nil
	}

	c.mu.Unlock()

	pipeline := NewAudioPipeline(cfg)
	pipeline.SetResampler(
		&LinearResampler{},
	)
	encoder, err :=
		EncoderFor(c.Codec)

	if err != nil {
		return err
	}

	decoder, err :=
		DecoderFor(c.Codec)

	if err != nil {
		return err
	}

	pipeline.SetEncoder(encoder)
	pipeline.SetDecoder(decoder)
	pipeline.SetResampler(
		&LinearResampler{},
	)
	pipeline.Start()

	c.mu.Lock()

	if c.closed {
		c.mu.Unlock()
		_ = pipeline.Close()
		return neurocall.ErrCallClosed
	}

	c.Pipeline = pipeline

	c.mu.Unlock()

	return nil
}
func (c *SIPCall) Speak(
	ctx context.Context,
	text string,
) error {

	c.mu.RLock()

	tts := c.TTS
	rtp := c.RTP
	pipeline := c.Pipeline
	codec := c.Codec
	closed := c.closed

	c.mu.RUnlock()

	if closed {
		return neurocall.ErrCallClosed
	}

	if tts == nil {
		return fmt.Errorf(
			"TTS is not configured",
		)
	}

	if rtp == nil {
		return fmt.Errorf(
			"RTP is not configured",
		)
	}

	if pipeline == nil {
		return fmt.Errorf(
			"audio pipeline is not configured",
		)
	}

	audio, err := tts.Synthesize(
		ctx,
		text,
	)
	if err != nil {
		return err
	}
	if audio.Format.Codec != "PCM16" {
		return fmt.Errorf(
			"unsupported TTS codec: %s",
			audio.Format.Codec,
		)
	}
	pcm := BytesToPCM16(audio.Data)
	if len(pcm) == 0 {
		return neurocall.ErrInvalidAudio
	}
	pcm = pipeline.Resample(
		pcm,
		audio.Format.SampleRate,
		codec.ClockRate,
	)
	encoder := pipeline.Encoder()
	if encoder == nil {
		return fmt.Errorf(
			"audio encoder is not configured",
		)
	}
	return rtp.WritePCMRealtime(
		ctx,
		encoder,
		pcm,
	)
}
func (c *SIPCall) CloseResources() error {
	c.mu.Lock()
	rtp := c.RTP
	pipeline := c.Pipeline
	tts := c.TTS
	c.RTP = nil
	c.Pipeline = nil
	c.TTS = nil
	c.closed = true
	c.mu.Unlock()
	var firstErr error
	if rtp != nil {
		if err := rtp.Close(); err != nil {
			firstErr = err
		}
	}
	if pipeline != nil {
		if err := pipeline.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if tts != nil {
		if err := tts.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
