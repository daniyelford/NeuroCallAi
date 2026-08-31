package core

func NewAudioEngine(
	pipeline *AudioPipeline,
) *AudioEngine {

	return &AudioEngine{
		pipeline: pipeline,
	}
}
func (e *AudioEngine) Start() error {

	e.mu.Lock()

	if e.running {
		e.mu.Unlock()
		return nil
	}

	e.running = true

	e.mu.Unlock()

	e.pipeline.Start()

	return nil
}
