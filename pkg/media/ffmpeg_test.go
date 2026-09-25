package media

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestNewFFmpeg(t *testing.T) {
	ffmpeg, err := NewFFmpeg("")
	if err != nil {
		t.Fatalf("NewFFmpeg failed: %v", err)
	}

	if ffmpeg.Path == "" {
		t.Fatal("ffmpeg path is empty")
	}

	t.Logf("ffmpeg path: %s", ffmpeg.Path)
}

func TestFFmpegInvalidArguments(t *testing.T) {
	ffmpeg := &FFmpeg{Path: "ffmpeg"}

	_, err := ffmpeg.ConvertToPCM16(
		context.Background(),
		"",
		16000,
		1,
	)

	if err != ErrInvalidInputPath {
		t.Fatalf("expected ErrInvalidInputPath, got %v", err)
	}
}

func TestFFmpegConvertToPCM16(t *testing.T) {
	projectRoot := filepath.Join("..", "..")

	inputPath := filepath.Join(
		projectRoot,
		"data",
		"raw",
		"commonvoice",
		"fa",
		"clips",
		"common_voice_fa_24958936.mp3",
	)

	if _, err := os.Stat(inputPath); err != nil {
		t.Skipf(
			"Common Voice fixture not available: %s",
			inputPath,
		)
	}

	ffmpeg, err := NewFFmpeg("")
	if err != nil {
		t.Fatalf("NewFFmpeg failed: %v", err)
	}

	data, err := ffmpeg.ConvertToPCM16(
		context.Background(),
		inputPath,
		16000,
		1,
	)
	if err != nil {
		t.Fatalf("ConvertToPCM16 failed: %v", err)
	}

	if len(data) == 0 {
		t.Fatal("PCM data is empty")
	}

	if len(data)%2 != 0 {
		t.Fatalf(
			"PCM16 data has odd byte length: %d",
			len(data),
		)
	}

	t.Logf(
		"converted %s -> %d PCM bytes",
		inputPath,
		len(data),
	)
}

func TestResolveFFmpegPath(t *testing.T) {
	path, err := ResolveFFmpegPath()
	if err != nil {
		t.Fatalf("ResolveFFmpegPath failed: %v", err)
	}

	if path == "" {
		t.Fatal("ffmpeg path is empty")
	}

	t.Logf("resolved ffmpeg: %s", path)
}
