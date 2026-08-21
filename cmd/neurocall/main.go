package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	// 	app := bootstrap.NewApp()
	// 	if err := app.Run(); err != nil {
	// 		log.Fatal(err)
	// 	}
	ctx, cancel := signal.NotifyContext(
		context.Background(),
		os.Interrupt,
		syscall.SIGTERM,
	)
	defer cancel()
	<-ctx.Done()
	// 	shutdownCtx, cancel := context.WithTimeout(
	// 		context.Background(),
	// 		10*time.Second,
	// 	)
	// 	defer cancel()
	// 	if err := app.Shutdown(shutdownCtx); err != nil {
	// 		log.Fatal(err)
	// 	}
}
