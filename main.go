package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/daniyelford/NeuroCallAi/core"
	"github.com/daniyelford/NeuroCallAi/pkg/llm"
	"github.com/daniyelford/NeuroCallAi/pkg/stt"
	"github.com/daniyelford/NeuroCallAi/pkg/tts"
)

func main() {
	cfg := core.DefaultConfig()
	vocabulary, err := stt.NewEnglishVocabulary()
	if err != nil {
		log.Fatal(err)
	}
	model := stt.NewSTTModel(
		80,
		128,
		vocabulary,
	)
	localSTT, err := stt.NewLocalSTT(
		model,
		stt.DefaultFeatureConfig(),
	)
	if err != nil {
		log.Fatal(err)
	}
	localLLM := llm.NewLocalLLM()
	localTTS := tts.NewLocalTTS()
	server := core.NewSIPServer(
		cfg.SIP,
		nil,
		nil,
		localSTT,
		localLLM,
		localTTS,
	)
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
