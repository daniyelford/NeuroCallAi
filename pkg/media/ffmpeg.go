package media

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
)

var (
	ErrInvalidFFmpeg       = errors.New("invalid ffmpeg")
	ErrFFmpegNotFound      = errors.New("ffmpeg executable not found")
	ErrInvalidInputPath    = errors.New("invalid input path")
	ErrInvalidSampleRate   = errors.New("invalid sample rate")
	ErrInvalidChannelCount = errors.New("invalid channel count")
	ErrFFmpegExecution     = errors.New("ffmpeg execution failed")
)

type FFmpeg struct {
	Path string
}

func NewFFmpeg(path string) (*FFmpeg, error) {
	if path == "" {
		resolved, err := ResolveFFmpegPath()
		if err != nil {
			return nil, err
		}

		path = resolved
	} else {
		resolved, err := exec.LookPath(path)
		if err != nil {
			return nil, fmt.Errorf(
				"%w: %s",
				ErrFFmpegNotFound,
				path,
			)
		}

		path = resolved
	}

	return &FFmpeg{
		Path: path,
	}, nil
}

func ResolveFFmpegPath() (string, error) {
	// 1. Search inside the project.
	if path := findProjectFFmpeg(); path != "" {
		return path, nil
	}

	// 2. Fallback to PATH.
	path, err := exec.LookPath("ffmpeg")
	if err == nil {
		return path, nil
	}

	return "", fmt.Errorf(
		"%w: ffmpeg was not found in project or PATH",
		ErrFFmpegNotFound,
	)
}

func findProjectFFmpeg() string {
	exeName := "ffmpeg"

	if runtime.GOOS == "windows" {
		exeName = "ffmpeg.exe"
	}

	dir, err := os.Getwd()
	if err != nil {
		return ""
	}

	for {
		candidate := filepath.Join(
			dir,
			"bin",
			"ffmpeg",
			exeName,
		)

		if fileExists(candidate) {
			return candidate
		}

		parent := filepath.Dir(dir)

		if parent == dir {
			break
		}

		dir = parent
	}

	return ""
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}

	return !info.IsDir()
}
func (f *FFmpeg) ConvertToPCM16(
	ctx context.Context,
	inputPath string,
	sampleRate int,
	channels int,
) ([]byte, error) {
	if f == nil || f.Path == "" {
		return nil, ErrInvalidFFmpeg
	}

	if inputPath == "" {
		return nil, ErrInvalidInputPath
	}

	if sampleRate <= 0 {
		return nil, ErrInvalidSampleRate
	}

	if channels <= 0 {
		return nil, ErrInvalidChannelCount
	}

	cmd := exec.CommandContext(
		ctx,
		f.Path,
		"-hide_banner",
		"-loglevel", "error",
		"-i", inputPath,
		"-f", "s16le",
		"-acodec", "pcm_s16le",
		"-ar", strconv.Itoa(sampleRate),
		"-ac", strconv.Itoa(channels),
		"pipe:1",
	)

	var stdout bytes.Buffer
	var stderr bytes.Buffer

	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		if stderr.Len() > 0 {
			return nil, fmt.Errorf(
				"%w: %s",
				ErrFFmpegExecution,
				stderr.String(),
			)
		}

		return nil, fmt.Errorf(
			"%w: %v",
			ErrFFmpegExecution,
			err,
		)
	}

	if stdout.Len() == 0 {
		return nil, ErrFFmpegExecution
	}

	return stdout.Bytes(), nil
}
