package core

func DefaultAudioReceiveConfig() AudioReceiveConfig {
	return AudioReceiveConfig{
		SampleRate: 8000,
		Channels:   1,
		FrameSize:  160,
	}
}
