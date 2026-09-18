package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/daniyelford/NeuroCallAi/core"
	"github.com/daniyelford/NeuroCallAi/pkg/openaipkg"
	"github.com/openai/openai-go/v3"
)

func main() {
	cfg := core.DefaultConfig()
	client := openai.NewClient()

	llm := openaipkg.NewLLM(
		&client,
		"gpt-6-astra",
	)

	stt := openaipkg.NewSTTProvider(
		&client,
	)
	tts := openaipkg.NewTTS(&client)
	server := core.NewSIPServer(
		cfg.SIP,
		nil,
		nil,
		stt,
		llm,
		tts,
	)
	// فعلاً STT واقعی نداریم.
	// FakeSTT فقط برای اینکه pipeline بتواند start شود.

	ctx, stop := signal.NotifyContext(
		context.Background(),
		os.Interrupt,
		syscall.SIGTERM,
	)
	defer stop()

	if err := server.Listen(ctx); err != nil {
		log.Fatalf("failed to start SIP server: %v", err)
	}

	log.Printf(
		"NeuroCallAi SIP server listening on %s:%d",
		cfg.SIP.ListenIP,
		cfg.SIP.SIPPort,
	)

	<-ctx.Done()

	log.Println("shutdown signal received")

	shutdownCtx, cancel := context.WithTimeout(
		context.Background(),
		10*time.Second,
	)
	defer cancel()

	done := make(chan error, 1)

	go func() {
		done <- server.Stop()
	}()

	select {
	case err := <-done:
		if err != nil {
			log.Printf("server shutdown error: %v", err)
		}
	case <-shutdownCtx.Done():
		log.Println("shutdown timeout exceeded")
	}

	log.Println("NeuroCallAi stopped")
}
