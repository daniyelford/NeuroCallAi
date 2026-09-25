package stt

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"math/rand"
	"os"
	"strings"
)

func WordErrorRate(
	reference string,
	hypothesis string,
) (float32, error) {

	referenceWords := strings.Fields(reference)
	hypothesisWords := strings.Fields(hypothesis)

	if len(referenceWords) == 0 {
		if len(hypothesisWords) == 0 {
			return 0, nil
		}

		return 0, ErrInvalidMetricInput
	}

	if len(hypothesisWords) == 0 {
		return 1, nil
	}

	previous := make([]int, len(hypothesisWords)+1)
	current := make([]int, len(hypothesisWords)+1)

	for j := 0; j <= len(hypothesisWords); j++ {
		previous[j] = j
	}

	for i := 1; i <= len(referenceWords); i++ {
		current[0] = i

		for j := 1; j <= len(hypothesisWords); j++ {
			cost := 0

			if referenceWords[i-1] != hypothesisWords[j-1] {
				cost = 1
			}

			deletion := previous[j] + 1
			insertion := current[j-1] + 1
			substitution := previous[j-1] + cost

			current[j] = minInt(
				deletion,
				insertion,
				substitution,
			)
		}

		previous, current = current, previous
	}

	distance := previous[len(hypothesisWords)]

	return float32(distance) /
		float32(len(referenceWords)), nil
}
func CharacterErrorRate(
	reference string,
	hypothesis string,
) (float32, error) {

	ref := []rune(reference)
	hyp := []rune(hypothesis)

	if len(ref) == 0 {
		if len(hyp) == 0 {
			return 0, nil
		}

		return 0, ErrInvalidMetricInput
	}

	if len(hyp) == 0 {
		return 1, nil
	}

	previous := make([]int, len(hyp)+1)
	current := make([]int, len(hyp)+1)

	for j := 0; j <= len(hyp); j++ {
		previous[j] = j
	}

	for i := 1; i <= len(ref); i++ {
		current[0] = i

		for j := 1; j <= len(hyp); j++ {
			cost := 0
			if ref[i-1] != hyp[j-1] {
				cost = 1
			}

			deletion := previous[j] + 1
			insertion := current[j-1] + 1
			substitution := previous[j-1] + cost

			current[j] = minInt(
				deletion,
				insertion,
				substitution,
			)
		}

		previous, current = current, previous
	}

	distance := previous[len(hyp)]

	return float32(distance) / float32(len(ref)), nil
}
func minInt(values ...int) int {
	result := values[0]

	for _, value := range values[1:] {
		if value < result {
			result = value
		}
	}

	return result
}
func LoadSTTModel(
	path string,
) (*STTModel, error) {

	if path == "" {
		return nil, ErrInvalidModelFile
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var file STTModelFile

	if err := json.Unmarshal(
		data,
		&file,
	); err != nil {
		return nil, err
	}

	if file.Version != 1 {
		return nil, ErrInvalidModelFile
	}

	if len(file.VocabularyTokens) == 0 {
		return nil, ErrInvalidModelFile
	}

	vocabulary, err := NewVocabulary(file.VocabularyTokens)
	if err != nil {
		return nil, err
	}

	if file.VocabularyBlankID != vocabulary.BlankID() {
		return nil, ErrInvalidModelFile
	}

	if file.VocabularyBlankID != vocabulary.BlankID() {
		return nil, ErrInvalidModelFile
	}

	if file.InputSize <= 0 ||
		file.HiddenSize <= 0 ||
		file.OutputSize <= 0 {
		return nil, ErrInvalidModelFile
	}

	model := NewSTTModel(
		file.InputSize,
		file.HiddenSize,
		vocabulary,
	)

	if model == nil {
		return nil, ErrInvalidModelFile
	}

	if len(model.Decoder.Weights) !=
		len(file.DecoderWeights) ||
		len(model.Decoder.Bias) !=
			len(file.DecoderBias) ||
		len(model.Encoder.ForwardLSTM.Cell.Weights) !=
			len(file.ForwardLSTMWeights) ||
		len(model.Encoder.ForwardLSTM.Cell.Bias) !=
			len(file.ForwardLSTMBias) ||
		len(model.Encoder.BackwardLSTM.Cell.Weights) !=
			len(file.BackwardLSTMWeights) ||
		len(model.Encoder.BackwardLSTM.Cell.Bias) !=
			len(file.BackwardLSTMBias) {
		return nil, ErrInvalidModelFile
	}

	copy(
		model.Decoder.Weights,
		file.DecoderWeights,
	)

	copy(
		model.Decoder.Bias,
		file.DecoderBias,
	)

	copy(
		model.Encoder.ForwardLSTM.Cell.Weights,
		file.ForwardLSTMWeights,
	)

	copy(
		model.Encoder.ForwardLSTM.Cell.Bias,
		file.ForwardLSTMBias,
	)

	copy(
		model.Encoder.BackwardLSTM.Cell.Weights,
		file.BackwardLSTMWeights,
	)

	copy(
		model.Encoder.BackwardLSTM.Cell.Bias,
		file.BackwardLSTMBias,
	)

	return model, nil
}
func SplitDataset(
	dataset *Dataset,
	trainRatio float32,
	validationRatio float32,
	testRatio float32,
	rng *rand.Rand,
) (*DatasetSplit, error) {
	if dataset == nil || dataset.Len() == 0 {
		return nil, ErrInvalidDatasetSplit
	}

	if trainRatio <= 0 ||
		validationRatio < 0 ||
		testRatio <= 0 {
		return nil, ErrInvalidDatasetSplit
	}

	totalRatio :=
		trainRatio +
			validationRatio +
			testRatio

	if totalRatio < 0.9999 ||
		totalRatio > 1.0001 {
		return nil, ErrInvalidDatasetSplit
	}

	if rng == nil {
		rng = rand.New(
			rand.NewSource(42),
		)
	}

	indices := make(
		[]int,
		dataset.Len(),
	)

	for i := range indices {
		indices[i] = i
	}

	rng.Shuffle(
		len(indices),
		func(i, j int) {
			indices[i], indices[j] =
				indices[j], indices[i]
		},
	)

	trainCount := int(
		float32(len(indices)) * trainRatio,
	)

	validationCount := int(
		float32(len(indices)) * validationRatio,
	)

	if trainCount <= 0 ||
		trainCount >= len(indices) {
		return nil, ErrInvalidDatasetSplit
	}

	validationStart := trainCount
	testStart := validationStart + validationCount

	if testStart >= len(indices) {
		return nil, ErrInvalidDatasetSplit
	}

	train := NewDataset()
	validation := NewDataset()
	test := NewDataset()

	for i, index := range indices {

		sample, err := dataset.Get(index)
		if err != nil {
			return nil, err
		}

		switch {
		case i < validationStart:
			if err := train.Add(*sample); err != nil {
				return nil, err
			}

		case i < testStart:
			if err := validation.Add(*sample); err != nil {
				return nil, err
			}

		default:
			if err := test.Add(*sample); err != nil {
				return nil, err
			}
		}
	}

	return &DatasetSplit{
		Train:      train,
		Validation: validation,
		Test:       test,
	}, nil
}
func frameToFFT(frame []float32, fftSize int) []Complex {
	input := make([]Complex, fftSize)

	limit := len(frame)

	if limit > fftSize {
		limit = fftSize
	}

	for i := 0; i < limit; i++ {
		input[i] = Complex{
			Real: float64(frame[i]),
		}
	}

	return input
}
func extractFrameFeature(
	frame []float32,
	window []float32,
	bank *MelFilterBank,
	fftSize int,
) []float32 {

	if len(frame) != len(window) {
		return nil
	}

	windowed := ApplyWindow(
		frame,
		window,
	)

	input := frameToFFT(
		windowed,
		fftSize,
	)

	spectrum := FFT(input)

	power := make(
		[]float32,
		fftSize/2+1,
	)

	for i := range power {
		real := spectrum[i].Real
		imag := spectrum[i].Imag

		power[i] = float32(
			(real*real + imag*imag) /
				float64(fftSize),
		)
	}

	mel := bank.Apply(power)

	return LogMel(mel)
}
func ExtractFeatures(
	samples []float32,
	config FeatureConfig,
) (*FeatureMatrix, error) {

	if err := config.Validate(); err != nil {
		return nil, err
	}

	if len(samples) == 0 {
		return nil, ErrInvalidAudio
	}

	samples = Normalize(samples)

	samples = PreEmphasis(
		samples,
		0.97,
	)

	frames := FrameAudio(
		samples,
		config.SampleRate,
		config.FrameDurationMs,
		config.HopDurationMs,
	)

	if len(frames) == 0 {
		return nil, ErrFeatureExtraction
	}

	frameSize := len(frames[0])

	window := HannWindow(frameSize)

	bank := NewMelFilterBank(
		config.SampleRate,
		config.FFTSize,
		config.MelBins,
		config.MinFrequency,
		config.MaxFrequency,
	)

	matrix := NewFeatureMatrix(
		len(frames),
		config.MelBins,
	)

	for i, frame := range frames {

		feature := extractFrameFeature(
			frame,
			window,
			bank,
			config.FFTSize,
		)

		if len(feature) != config.MelBins {
			return nil, ErrFeatureExtraction
		}

		for j, value := range feature {
			matrix.Set(
				i,
				j,
				value,
			)
		}
	}

	return matrix, nil
}
func DefaultFeatureConfig() FeatureConfig {
	return FeatureConfig{
		SampleRate: 16000,

		FrameDurationMs: 25,
		HopDurationMs:   10,

		FFTSize: 512,

		MelBins: 80,

		MinFrequency: 0,
		MaxFrequency: 8000,
	}
}
func LogMel(mel []float32) []float32 {
	result := make(
		[]float32,
		len(mel),
	)

	const epsilon = 1e-10

	for i, value := range mel {
		result[i] = float32(
			math.Log(
				float64(value) + epsilon,
			),
		)
	}

	return result
}
func PCM16ToFloat32(data []byte) ([]float32, error) {
	if len(data) == 0 {
		return nil, ErrInvalidAudio
	}

	if len(data)%2 != 0 {
		return nil, ErrInvalidAudio
	}

	samples := make([]float32, len(data)/2)

	for i := 0; i < len(samples); i++ {
		value := int16(
			binary.LittleEndian.Uint16(
				data[i*2 : i*2+2],
			),
		)

		samples[i] = float32(value) / 32768.0
	}

	return samples, nil
}
func Normalize(samples []float32) []float32 {
	if len(samples) == 0 {
		return nil
	}

	var max float32

	for _, sample := range samples {
		value := float32(math.Abs(float64(sample)))

		if value > max {
			max = value
		}
	}

	if max == 0 {
		return append([]float32(nil), samples...)
	}

	result := make([]float32, len(samples))

	for i, sample := range samples {
		result[i] = sample / max
	}

	return result
}
func encodePCM16(values ...int16) []byte {
	data := make([]byte, len(values)*2)

	for i, value := range values {
		binary.LittleEndian.PutUint16(
			data[i*2:i*2+2],
			uint16(value),
		)
	}

	return data
}
func FFT(input []Complex) []Complex {
	n := len(input)

	if n == 0 {
		return nil
	}

	if n == 1 {
		return []Complex{
			input[0],
		}
	}

	if n%2 != 0 {
		return DFT(input)
	}

	even := make([]Complex, n/2)
	odd := make([]Complex, n/2)

	for i := 0; i < n/2; i++ {
		even[i] = input[2*i]
		odd[i] = input[2*i+1]
	}

	even = FFT(even)
	odd = FFT(odd)

	result := make([]Complex, n)

	for k := 0; k < n/2; k++ {
		angle := -2 * math.Pi * float64(k) / float64(n)

		w := Complex{
			Real: math.Cos(angle),
			Imag: math.Sin(angle),
		}

		t := multiply(w, odd[k])

		result[k] = add(
			even[k],
			t,
		)

		result[k+n/2] = subtract(
			even[k],
			t,
		)
	}

	return result
}
func DFT(input []Complex) []Complex {
	n := len(input)

	result := make([]Complex, n)

	for k := 0; k < n; k++ {
		for j := 0; j < n; j++ {
			angle := -2 * math.Pi *
				float64(k*j) /
				float64(n)

			c := Complex{
				Real: math.Cos(angle),
				Imag: math.Sin(angle),
			}

			result[k] = add(
				result[k],
				multiply(input[j], c),
			)
		}
	}

	return result
}
func add(a, b Complex) Complex {
	return Complex{
		Real: a.Real + b.Real,
		Imag: a.Imag + b.Imag,
	}
}
func subtract(a, b Complex) Complex {
	return Complex{
		Real: a.Real - b.Real,
		Imag: a.Imag - b.Imag,
	}
}
func multiply(a, b Complex) Complex {
	return Complex{
		Real: a.Real*b.Real - a.Imag*b.Imag,
		Imag: a.Real*b.Imag + a.Imag*b.Real,
	}
}
func PowerSpectrum(frame []float32) []float32 {
	if len(frame) == 0 {
		return nil
	}

	input := make([]Complex, len(frame))

	for i, sample := range frame {
		input[i] = Complex{
			Real: float64(sample),
		}
	}

	spectrum := FFT(input)

	size := len(spectrum)/2 + 1

	power := make([]float32, size)

	for i := 0; i < size; i++ {
		real := spectrum[i].Real
		imag := spectrum[i].Imag

		power[i] = float32(
			(real*real + imag*imag) /
				float64(len(frame)),
		)
	}

	return power
}
func HertzToMel(hz float64) float64 {
	return 2595 *
		math.Log10(
			1+hz/700,
		)
}
func MelToHertz(mel float64) float64 {
	return 700 *
		(math.Pow(
			10,
			mel/2595,
		) - 1)
}
func PreEmphasis(samples []float32, coefficient float32) []float32 {
	if len(samples) == 0 {
		return nil
	}

	if coefficient < 0 || coefficient > 1 {
		coefficient = 0.97
	}

	result := make([]float32, len(samples))

	result[0] = samples[0]

	for i := 1; i < len(samples); i++ {
		result[i] = samples[i] - coefficient*samples[i-1]
	}

	return result
}
func FrameAudio(
	samples []float32,
	sampleRate int,
	frameDurationMs float64,
	hopDurationMs float64,
) [][]float32 {

	if len(samples) == 0 || sampleRate <= 0 {
		return nil
	}

	frameSize := int(
		float64(sampleRate) * frameDurationMs / 1000,
	)

	hopSize := int(
		float64(sampleRate) * hopDurationMs / 1000,
	)

	if frameSize <= 0 || hopSize <= 0 {
		return nil
	}

	if len(samples) < frameSize {
		frame := make([]float32, frameSize)
		copy(frame, samples)

		return [][]float32{frame}
	}

	frameCount := 1 + (len(samples)-frameSize)/hopSize

	frames := make([][]float32, frameCount)

	for i := 0; i < frameCount; i++ {
		start := i * hopSize

		frame := make([]float32, frameSize)

		copy(
			frame,
			samples[start:start+frameSize],
		)

		frames[i] = frame
	}

	return frames
}
func HannWindow(size int) []float32 {
	if size <= 0 {
		return nil
	}

	window := make([]float32, size)

	for i := 0; i < size; i++ {
		window[i] = float32(
			0.5 * (1 -
				math.Cos(
					2*math.Pi*float64(i)/float64(size-1),
				)),
		)
	}

	return window
}
func ApplyWindow(
	frame []float32,
	window []float32,
) []float32 {

	if len(frame) != len(window) {
		return nil
	}

	result := make([]float32, len(frame))

	for i := range frame {
		result[i] = frame[i] * window[i]
	}

	return result
}
func PrepareFrames(
	samples []float32,
	sampleRate int,
) [][]float32 {

	if len(samples) == 0 {
		return nil
	}

	samples = Normalize(samples)

	samples = PreEmphasis(
		samples,
		0.97,
	)

	frames := FrameAudio(
		samples,
		sampleRate,
		25,
		10,
	)

	window := HannWindow(
		int(float64(sampleRate) * 25 / 1000),
	)

	for i := range frames {
		frames[i] = ApplyWindow(
			frames[i],
			window,
		)
	}

	return frames
}
func PCM16ToWAV(
	pcm []int16,
	sampleRate int,
	channels int,
) []byte {

	var buf bytes.Buffer

	dataSize := len(pcm) * 2

	buf.Write([]byte("RIFF"))

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint32(36+dataSize),
	)

	buf.Write([]byte("WAVE"))

	buf.Write([]byte("fmt "))

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint32(16),
	)

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint16(1),
	)

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint16(channels),
	)

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint32(sampleRate),
	)

	byteRate :=
		sampleRate *
			channels *
			2

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint32(byteRate),
	)

	blockAlign :=
		uint16(channels * 2)

	binary.Write(
		&buf,
		binary.LittleEndian,
		blockAlign,
	)

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint16(16),
	)

	buf.Write([]byte("data"))

	binary.Write(
		&buf,
		binary.LittleEndian,
		uint32(dataSize),
	)

	for _, sample := range pcm {
		binary.Write(
			&buf,
			binary.LittleEndian,
			sample,
		)
	}

	return buf.Bytes()
}
func LoadWAV(path string) (*Audio, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	header := make([]byte, 12)

	if _, err := io.ReadFull(file, header); err != nil {
		return nil, ErrInvalidWAV
	}

	if string(header[0:4]) != "RIFF" ||
		string(header[8:12]) != "WAVE" {
		return nil, ErrInvalidWAV
	}

	var (
		audioFormat   uint16
		numChannels   uint16
		sampleRate    uint32
		bitsPerSample uint16
		data          []byte
	)

	for {
		chunkHeader := make([]byte, 8)

		if _, err := io.ReadFull(file, chunkHeader); err != nil {
			if err == io.EOF {
				break
			}

			return nil, ErrInvalidWAV
		}

		chunkID := string(chunkHeader[0:4])
		chunkSize := binary.LittleEndian.Uint32(chunkHeader[4:8])

		switch chunkID {
		case "fmt ":
			if chunkSize < 16 {
				return nil, ErrInvalidWAV
			}

			fmtData := make([]byte, chunkSize)

			if _, err := io.ReadFull(file, fmtData); err != nil {
				return nil, ErrInvalidWAV
			}

			audioFormat = binary.LittleEndian.Uint16(fmtData[0:2])
			numChannels = binary.LittleEndian.Uint16(fmtData[2:4])
			sampleRate = binary.LittleEndian.Uint32(fmtData[4:8])
			bitsPerSample = binary.LittleEndian.Uint16(fmtData[14:16])

		case "data":
			data = make([]byte, chunkSize)

			if _, err := io.ReadFull(file, data); err != nil {
				return nil, ErrInvalidWAV
			}

		default:
			if _, err := file.Seek(int64(chunkSize), io.SeekCurrent); err != nil {
				return nil, ErrInvalidWAV
			}
		}

		if len(data) > 0 {
			break
		}
	}

	if audioFormat != 1 {
		return nil, ErrUnsupportedWAV
	}

	if numChannels == 0 || sampleRate == 0 {
		return nil, ErrInvalidWAV
	}

	if bitsPerSample != 16 {
		return nil, ErrUnsupportedWAV
	}

	samples, err := PCM16ToFloat32(data)
	if err != nil {
		return nil, err
	}

	return NewAudio(
		samples,
		int(sampleRate),
		int(numChannels),
	)
}
func LoadManifest(path string) ([]ManifestEntry, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	entries := make([]ManifestEntry, 0)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if line == "" {
			continue
		}

		parts := strings.SplitN(line, "\t", 2)

		if len(parts) != 2 {
			return nil, ErrManifestLine
		}

		audioPath := strings.TrimSpace(parts[0])
		text := strings.TrimSpace(parts[1])

		if audioPath == "" || text == "" {
			return nil, ErrManifestLine
		}

		entries = append(entries, ManifestEntry{
			AudioPath: audioPath,
			Text:      text,
		})
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	if len(entries) == 0 {
		return nil, ErrInvalidManifest
	}

	return entries, nil
}
func logSumExp(values ...float64) float64 {
	if len(values) == 0 {
		return math.Inf(-1)
	}

	maxValue := values[0]

	for _, value := range values[1:] {
		if value > maxValue {
			maxValue = value
		}
	}

	if math.IsInf(maxValue, -1) {
		return maxValue
	}

	var sum float64

	for _, value := range values {
		sum += math.Exp(value - maxValue)
	}

	return maxValue + math.Log(sum)
}
func logSoftmax(
	values []float32,
) []float64 {
	if len(values) == 0 {
		return nil
	}

	maxValue := float64(values[0])

	for _, value := range values[1:] {
		if float64(value) > maxValue {
			maxValue = float64(value)
		}
	}

	var sum float64

	for _, value := range values {
		sum += math.Exp(
			float64(value) - maxValue,
		)
	}

	logSum := maxValue + math.Log(sum)

	result := make([]float64, len(values))

	for i, value := range values {
		result[i] =
			float64(value) - logSum
	}

	return result
}
func extendCTCTarget(
	target []int,
	blankID int,
) []int {

	extended := make([]int, 0, len(target)*2+1)

	extended = append(
		extended,
		blankID,
	)

	for _, token := range target {
		extended = append(
			extended,
			token,
			blankID,
		)
	}

	return extended
}
func CTCLoss(
	logits *Tensor3D,
	target []int,
	inputLength int,
	blankID int,
) (float32, error) {

	if logits == nil {
		return 0, ErrInvalidCTCLoss
	}

	if logits.Batch != 1 {
		return 0, ErrInvalidCTCLoss
	}

	if inputLength <= 0 ||
		inputLength > logits.Time {
		return 0, ErrInvalidCTCLoss
	}

	if blankID < 0 ||
		blankID >= logits.Features {
		return 0, ErrInvalidCTCLoss
	}

	extended := extendCTCTarget(
		target,
		blankID,
	)

	states := len(extended)

	if states == 0 {
		return 0, ErrInvalidCTCLoss
	}

	// alpha[t][s]
	alpha := make(
		[][]float64,
		inputLength,
	)

	for t := 0; t < inputLength; t++ {
		alpha[t] = make(
			[]float64,
			states,
		)

		for s := range alpha[t] {
			alpha[t][s] = math.Inf(-1)
		}
	}

	// t = 0
	// first, err := logits.Get(
	// 	0,
	// 	0,
	// 	extended[0],
	// )

	// if err != nil {
	// 	return 0, err
	// }

	// firstLogProbs := logSoftmax(
	// 	[]float32{first},
	// )

	// _ = firstLogProbs

	// Calculate complete log-softmax
	// for timestep zero.
	row := make([]float32, logits.Features)

	for c := 0; c < logits.Features; c++ {
		value, err := logits.Get(
			0,
			0,
			c,
		)

		if err != nil {
			return 0, err
		}

		row[c] = value
	}

	logProbs := logSoftmax(row)

	alpha[0][0] = logProbs[extended[0]]

	if states > 1 {
		alpha[0][1] =
			logProbs[extended[1]]
	}

	for t := 1; t < inputLength; t++ {
		row := make([]float32, logits.Features)

		for c := 0; c < logits.Features; c++ {
			value, err := logits.Get(
				0,
				t,
				c,
			)

			if err != nil {
				return 0, err
			}

			row[c] = value
		}

		logProbs := logSoftmax(row)

		for s := 0; s < states; s++ {
			current := extended[s]

			values := []float64{
				alpha[t-1][s],
			}

			if s > 0 {
				values = append(
					values,
					alpha[t-1][s-1],
				)
			}

			if s > 1 &&
				current != blankID &&
				current != extended[s-2] {

				values = append(
					values,
					alpha[t-1][s-2],
				)
			}

			alpha[t][s] =
				logSumExp(values...) +
					logProbs[current]
		}
	}

	last := alpha[inputLength-1]

	logLikelihood := last[states-1]

	if states > 1 {
		logLikelihood = logSumExp(
			last[states-1],
			last[states-2],
		)
	}

	loss := -logLikelihood

	return float32(loss), nil
}
func CTCLossBatch(
	logits *Tensor3D,
	targets []int,
	inputLengths []int,
	targetLengths []int,
	blankID int,
) (float32, error) {

	if logits == nil {
		return 0, ErrInvalidCTCBatchLoss
	}

	if len(inputLengths) != logits.Batch ||
		len(targetLengths) != logits.Batch {
		return 0, ErrInvalidCTCBatchLoss
	}

	if blankID < 0 || blankID >= logits.Features {
		return 0, ErrInvalidCTCBatchLoss
	}

	targetOffset := 0
	var totalLoss float32

	for batch := 0; batch < logits.Batch; batch++ {
		inputLength := inputLengths[batch]
		targetLength := targetLengths[batch]

		if inputLength <= 0 ||
			inputLength > logits.Time {
			return 0, ErrInvalidCTCBatchLoss
		}

		if targetLength < 0 {
			return 0, ErrInvalidCTCBatchLoss
		}

		if targetOffset+targetLength > len(targets) {
			return 0, ErrInvalidCTCBatchLoss
		}
		target := targets[targetOffset : targetOffset+targetLength]
		// CTCLoss currently accepts batch size 1.
		sampleLogits := NewTensor3D(
			1,
			logits.Time,
			logits.Features,
		)

		if sampleLogits == nil {
			return 0, ErrInvalidCTCBatchLoss
		}

		for time := 0; time < inputLength; time++ {
			for feature := 0; feature < logits.Features; feature++ {
				value, err := logits.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return 0, err
				}

				if err := sampleLogits.Set(
					0,
					time,
					feature,
					value,
				); err != nil {
					return 0, err
				}
			}
		}

		loss, err := CTCLoss(
			sampleLogits,
			target,
			inputLength,
			blankID,
		)

		if err != nil {
			return 0, err
		}

		totalLoss += loss
		targetOffset += targetLength
	}

	if targetOffset != len(targets) {
		return 0, ErrInvalidCTCBatchLoss
	}

	if logits.Batch == 0 {
		return 0, ErrInvalidCTCBatchLoss
	}

	return totalLoss / float32(logits.Batch), nil
}
func CTCGradient(
	logits *Tensor3D,
	target []int,
	inputLength int,
	blankID int,
) (*Tensor3D, error) {

	if logits == nil {
		return nil, ErrInvalidCTCGradient
	}

	if logits.Batch != 1 {
		return nil, ErrInvalidCTCGradient
	}

	if inputLength <= 0 ||
		inputLength > logits.Time {
		return nil, ErrInvalidCTCGradient
	}

	if blankID < 0 ||
		blankID >= logits.Features {
		return nil, ErrInvalidCTCGradient
	}

	for _, token := range target {
		if token < 0 || token >= logits.Features {
			return nil, ErrInvalidCTCGradient
		}
	}

	extended := extendCTCTarget(
		target,
		blankID,
	)

	states := len(extended)

	if states == 0 {
		return nil, ErrInvalidCTCGradient
	}

	// ------------------------------------------------------------
	// Log probabilities
	// ------------------------------------------------------------

	logProbs := make(
		[][]float64,
		inputLength,
	)

	for t := 0; t < inputLength; t++ {
		row := make([]float32, logits.Features)

		for c := 0; c < logits.Features; c++ {
			value, err := logits.Get(
				0,
				t,
				c,
			)

			if err != nil {
				return nil, err
			}

			row[c] = value
		}

		logProbs[t] = logSoftmax(row)
	}

	// ------------------------------------------------------------
	// Forward variables
	// ------------------------------------------------------------

	alpha := make(
		[][]float64,
		inputLength,
	)

	for t := 0; t < inputLength; t++ {
		alpha[t] = make(
			[]float64,
			states,
		)

		for s := 0; s < states; s++ {
			alpha[t][s] = math.Inf(-1)
		}
	}

	alpha[0][0] =
		logProbs[0][extended[0]]

	if states > 1 {
		alpha[0][1] =
			logProbs[0][extended[1]]
	}

	for t := 1; t < inputLength; t++ {
		for s := 0; s < states; s++ {
			current := extended[s]

			values := []float64{
				alpha[t-1][s],
			}

			if s > 0 {
				values = append(
					values,
					alpha[t-1][s-1],
				)
			}

			if s > 1 &&
				current != blankID &&
				current != extended[s-2] {

				values = append(
					values,
					alpha[t-1][s-2],
				)
			}

			alpha[t][s] =
				logSumExp(values...) +
					logProbs[t][current]
		}
	}

	// ------------------------------------------------------------
	// Total log likelihood
	// ------------------------------------------------------------

	last := alpha[inputLength-1]

	logLikelihood := last[states-1]

	if states > 1 {
		logLikelihood = logSumExp(
			last[states-1],
			last[states-2],
		)
	}

	if math.IsInf(logLikelihood, -1) ||
		math.IsNaN(logLikelihood) {
		return nil, ErrInvalidCTCGradient
	}

	// ------------------------------------------------------------
	// Backward variables
	//
	// beta[t][s] does NOT include the emission probability
	// at timestep t.
	// ------------------------------------------------------------

	beta := make(
		[][]float64,
		inputLength,
	)

	for t := 0; t < inputLength; t++ {
		beta[t] = make(
			[]float64,
			states,
		)

		for s := 0; s < states; s++ {
			beta[t][s] = math.Inf(-1)
		}
	}

	beta[inputLength-1][states-1] = 0

	if states > 1 {
		beta[inputLength-1][states-2] = 0
	}

	for t := inputLength - 2; t >= 0; t-- {
		for s := 0; s < states; s++ {

			values := []float64{
				beta[t+1][s] +
					logProbs[t+1][extended[s]],
			}

			if s+1 < states {
				values = append(
					values,
					beta[t+1][s+1]+
						logProbs[t+1][extended[s+1]],
				)
			}

			if s+2 < states &&
				extended[s] != blankID &&
				extended[s] != extended[s+2] {

				values = append(
					values,
					beta[t+1][s+2]+
						logProbs[t+1][extended[s+2]],
				)
			}

			beta[t][s] =
				logSumExp(values...)
		}
	}

	// ------------------------------------------------------------
	// Gradient
	//
	// dL/dz = softmax(z) - posterior
	// ------------------------------------------------------------

	gradient := NewTensor3D(
		1,
		logits.Time,
		logits.Features,
	)

	if gradient == nil {
		return nil, ErrInvalidCTCGradient
	}

	for t := 0; t < inputLength; t++ {

		posterior := make(
			[]float64,
			logits.Features,
		)

		for s := 0; s < states; s++ {

			if math.IsInf(alpha[t][s], -1) ||
				math.IsInf(beta[t][s], -1) {
				continue
			}

			logPosterior :=
				alpha[t][s] +
					beta[t][s] -
					logLikelihood

			value := math.Exp(logPosterior)

			if math.IsNaN(value) ||
				math.IsInf(value, 0) {
				return nil, ErrInvalidCTCGradient
			}

			token := extended[s]

			posterior[token] += value
		}

		for token := 0; token < logits.Features; token++ {
			probability :=
				math.Exp(logProbs[t][token])

			value :=
				float32(
					probability -
						posterior[token],
				)

			if math.IsNaN(float64(value)) ||
				math.IsInf(float64(value), 0) {
				return nil, ErrInvalidCTCGradient
			}

			if err := gradient.Set(
				0,
				t,
				token,
				value,
			); err != nil {
				return nil, err
			}
		}
	}

	return gradient, nil
}
func featureMatrixToTensor(
	features *FeatureMatrix,
) *Tensor3D {

	if features == nil ||
		features.Rows <= 0 ||
		features.Cols <= 0 {
		return nil
	}

	tensor := NewTensor3D(
		1,
		features.Rows,
		features.Cols,
	)

	for time := 0; time < features.Rows; time++ {
		for feature := 0; feature < features.Cols; feature++ {

			value := features.Get(
				time,
				feature,
			)

			if err := tensor.Set(
				0,
				time,
				feature,
				value,
			); err != nil {
				return nil
			}
		}
	}

	return tensor
}
func LoadTrainingCheckpoint(path string) (*TrainingCheckpoint, error) {
	if strings.TrimSpace(path) == "" {
		return nil, ErrInvalidTrainingCheckpoint
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var checkpoint TrainingCheckpoint

	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return nil, err
	}

	if (checkpoint.Version != 1 &&
		checkpoint.Version != 2) ||
		checkpoint.Model == nil ||
		checkpoint.Optimizer == nil {
		return nil, ErrInvalidTrainingCheckpoint
	}

	return &checkpoint, nil
}
func SaveTrainingCheckpoint(
	path string,
	checkpoint *TrainingCheckpoint,
) error {

	if path == "" ||
		checkpoint == nil {
		return ErrInvalidTrainingCheckpoint
	}

	if (checkpoint.Version != 1 &&
		checkpoint.Version != 2) ||
		checkpoint.Model == nil ||
		checkpoint.Optimizer == nil {
		return ErrInvalidTrainingCheckpoint
	}

	data, err := json.MarshalIndent(
		checkpoint,
		"",
		"  ",
	)
	if err != nil {
		return err
	}

	return os.WriteFile(
		path,
		data,
		0644,
	)
}
func XavierUniform(
	weights []float32,
	fanIn int,
	fanOut int,
	rng *rand.Rand,
) {
	if len(weights) == 0 ||
		fanIn <= 0 ||
		fanOut <= 0 ||
		rng == nil {
		return
	}

	limit := float64(math.Sqrt(
		6.0 / float64(fanIn+fanOut),
	))

	for i := range weights {
		weights[i] = float32(
			(rng.Float64()*2.0 - 1.0) * limit,
		)
	}
}
func ZeroInitialize(values []float32) {
	for i := range values {
		values[i] = 0
	}
}
func InitializeLinear(
	layer *Linear,
	rng *rand.Rand,
) {
	if layer == nil {
		return
	}

	XavierUniform(
		layer.Weights,
		layer.InFeatures,
		layer.OutFeatures,
		rng,
	)

	ZeroInitialize(layer.Bias)
}
func InitializeLSTMCell(
	cell *LSTMCell,
	rng *rand.Rand,
) {
	if cell == nil {
		return
	}

	XavierUniform(
		cell.Weights,
		cell.InputSize+cell.HiddenSize,
		cell.HiddenSize*4,
		rng,
	)

	ZeroInitialize(cell.Bias)
}
func NormalizeAudio(
	audio *Audio,
	targetSampleRate int,
) (*Audio, error) {
	if audio == nil {
		return nil, ErrInvalidAudio
	}

	if len(audio.Samples) == 0 ||
		audio.SampleRate <= 0 ||
		audio.Channels <= 0 ||
		targetSampleRate <= 0 {
		return nil, ErrInvalidAudio
	}

	samples := audio.Samples

	// Convert to mono.
	if audio.Channels > 1 {
		if len(samples)%audio.Channels != 0 {
			return nil, ErrInvalidAudio
		}

		monoSamples := make([]float32, len(samples)/audio.Channels)

		for i := range monoSamples {
			var sum float32

			for channel := 0; channel < audio.Channels; channel++ {
				sum += samples[i*audio.Channels+channel]
			}

			monoSamples[i] =
				sum / float32(audio.Channels)
		}

		samples = monoSamples
	}

	// Already at target sample rate.
	if audio.SampleRate == targetSampleRate {
		return NewAudio(
			samples,
			targetSampleRate,
			1,
		)
	}

	resampled := resampleLinear(
		samples,
		audio.SampleRate,
		targetSampleRate,
	)

	if len(resampled) == 0 {
		return nil, ErrInvalidAudio
	}

	return NewAudio(
		resampled,
		targetSampleRate,
		1,
	)
}

func resampleLinear(
	input []float32,
	inputRate int,
	outputRate int,
) []float32 {
	if len(input) == 0 ||
		inputRate <= 0 ||
		outputRate <= 0 {
		return nil
	}

	if inputRate == outputRate {
		return append([]float32(nil), input...)
	}

	outputLength := int(
		math.Round(
			float64(len(input)) *
				float64(outputRate) /
				float64(inputRate),
		),
	)

	if outputLength <= 0 {
		return nil
	}

	output := make([]float32, outputLength)

	ratio :=
		float64(inputRate) /
			float64(outputRate)

	for i := 0; i < outputLength; i++ {
		position := float64(i) * ratio

		left := int(math.Floor(position))
		right := left + 1

		if left >= len(input) {
			left = len(input) - 1
		}

		if right >= len(input) {
			right = len(input) - 1
		}

		fraction := float32(position - math.Floor(position))

		output[i] =
			input[left]*(1-fraction) +
				input[right]*fraction
	}

	return output
}
func NewTokenizerFromTexts(texts []string) (*Tokenizer, error) {
	vocabulary, err := NewVocabularyFromTexts(texts)
	if err != nil {
		return nil, err
	}

	return NewTokenizer(vocabulary)
}
