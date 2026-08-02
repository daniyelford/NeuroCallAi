package logger

import (
	"log/slog"
	"os"

	"github.com/daniyelford/neurocall/pkg/neurocall"
)

type Console struct {
	log *slog.Logger
}

func NewConsole() *Console {
	return &Console{
		log: slog.New(
			slog.NewTextHandler(
				os.Stdout,
				&slog.HandlerOptions{
					Level: slog.LevelDebug,
				},
			),
		),
	}
}

func (l *Console) Debug(msg string, args ...any) {
	l.log.Debug(msg, args...)
}

func (l *Console) Info(msg string, args ...any) {
	l.log.Info(msg, args...)
}

func (l *Console) Warn(msg string, args ...any) {
	l.log.Warn(msg, args...)
}

func (l *Console) Error(msg string, args ...any) {
	l.log.Error(msg, args...)
}

var _ neurocall.Logger = (*Console)(nil)
