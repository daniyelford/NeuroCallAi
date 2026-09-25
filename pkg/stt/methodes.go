package stt

import (
	"bufio"
	"context"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/daniyelford/NeuroCallAi/pkg/media"
	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewAudio(
	samples []float32,
	sampleRate int,
	channels int,
) (*Audio, error) {

	if len(samples) == 0 {
		return nil, ErrInvalidAudio
	}

	if sampleRate <= 0 {
		return nil, ErrInvalidAudio
	}

	if channels <= 0 {
		return nil, ErrInvalidAudio
	}

	return &Audio{
		Samples:    samples,
		SampleRate: sampleRate,
		Channels:   channels,
	}, nil
}
func NewMelFilterBank(
	sampleRate int,
	fftSize int,
	numMels int,
	minFreq float64,
	maxFreq float64,
) *MelFilterBank {

	minMel := HertzToMel(minFreq)
	maxMel := HertzToMel(maxFreq)

	points := make([]float64, numMels+2)

	for i := range points {
		points[i] =
			minMel +
				float64(i)*
					(maxMel-minMel)/
					float64(numMels+1)
	}

	frequencies := make([]float64, len(points))

	for i, mel := range points {
		frequencies[i] = MelToHertz(mel)
	}

	bins := make([]int, len(frequencies))

	for i, freq := range frequencies {
		bins[i] = int(
			math.Floor(
				float64(fftSize+1) *
					freq /
					float64(sampleRate),
			),
		)
	}

	filters := make([][]float32, numMels)

	numBins := fftSize/2 + 1

	for m := 0; m < numMels; m++ {
		filter := make([]float32, numBins)

		left := bins[m]
		center := bins[m+1]
		right := bins[m+2]

		for k := left; k < center && k < numBins; k++ {
			if center != left {
				filter[k] =
					float32(k-left) /
						float32(center-left)
			}
		}

		for k := center; k < right && k < numBins; k++ {
			if right != center {
				filter[k] =
					float32(right-k) /
						float32(right-center)
			}
		}

		filters[m] = filter
	}

	return &MelFilterBank{
		Filters: filters,
	}
}
func (bank *MelFilterBank) Apply(
	power []float32,
) []float32 {

	if len(power) == 0 {
		return nil
	}

	result := make(
		[]float32,
		len(bank.Filters),
	)

	for i, filter := range bank.Filters {
		var sum float32

		limit := len(power)

		if len(filter) < limit {
			limit = len(filter)
		}

		for k := 0; k < limit; k++ {
			sum += power[k] * filter[k]
		}

		result[i] = sum
	}

	return result
}
func (c FeatureConfig) Validate() error {
	if c.SampleRate <= 0 {
		return ErrInvalidFeatureConfig
	}

	if c.FrameDurationMs <= 0 {
		return ErrInvalidFeatureConfig
	}

	if c.HopDurationMs <= 0 {
		return ErrInvalidFeatureConfig
	}

	if c.FFTSize <= 0 {
		return ErrInvalidFeatureConfig
	}

	if c.MelBins <= 0 {
		return ErrInvalidFeatureConfig
	}

	if c.MinFrequency < 0 {
		return ErrInvalidFeatureConfig
	}

	if c.MaxFrequency <= c.MinFrequency {
		return ErrInvalidFeatureConfig
	}

	if c.MaxFrequency > float64(c.SampleRate)/2 {
		return ErrInvalidFeatureConfig
	}

	return nil
}

// FeatureMatrix
func NewFeatureMatrix(rows, cols int) *FeatureMatrix {
	if rows <= 0 || cols <= 0 {
		return nil
	}

	return &FeatureMatrix{
		Data: make([]float32, rows*cols),
		Rows: rows,
		Cols: cols,
	}
}
func (m *FeatureMatrix) Set(row, col int, value float32) {
	if m == nil {
		return
	}

	if row < 0 || row >= m.Rows {
		return
	}

	if col < 0 || col >= m.Cols {
		return
	}

	m.Data[row*m.Cols+col] = value
}
func (m *FeatureMatrix) Get(row, col int) float32 {
	if m == nil {
		return 0
	}

	if row < 0 || row >= m.Rows {
		return 0
	}

	if col < 0 || col >= m.Cols {
		return 0
	}

	return m.Data[row*m.Cols+col]
}

// FeatureMatrix
func NewVocabulary(tokens []string) (*Vocabulary, error) {
	if len(tokens) == 0 {
		return nil, ErrEmptyVocabulary
	}

	tokenToID := make(map[string]int, len(tokens))

	for i, token := range tokens {
		if token == "" {
			return nil, ErrInvalidToken
		}

		if _, exists := tokenToID[token]; exists {
			return nil, ErrInvalidToken
		}

		tokenToID[token] = i
	}

	blankID, ok := tokenToID[BlankToken]
	if !ok {
		return nil, ErrInvalidToken
	}

	return &Vocabulary{
		tokens:    append([]string(nil), tokens...),
		tokenToID: tokenToID,
		blankID:   blankID,
	}, nil
}
func (v *Vocabulary) Size() int {
	if v == nil {
		return 0
	}

	return len(v.tokens)
}
func (v *Vocabulary) BlankID() int {
	if v == nil {
		return -1
	}

	return v.blankID
}
func (v *Vocabulary) EncodeToken(token string) (int, error) {
	if v == nil {
		return -1, ErrEmptyVocabulary
	}

	id, ok := v.tokenToID[token]
	if !ok {
		return -1, ErrUnknownToken
	}

	return id, nil
}
func (v *Vocabulary) DecodeID(id int) (string, error) {
	if v == nil {
		return "", ErrEmptyVocabulary
	}

	if id < 0 || id >= len(v.tokens) {
		return "", ErrUnknownToken
	}

	return v.tokens[id], nil
}
func NewEnglishVocabulary() (*Vocabulary, error) {
	tokens := []string{
		BlankToken,

		"a",
		"b",
		"c",
		"d",
		"e",
		"f",
		"g",
		"h",
		"i",
		"j",
		"k",
		"l",
		"m",
		"n",
		"o",
		"p",
		"q",
		"r",
		"s",
		"t",
		"u",
		"v",
		"w",
		"x",
		"y",
		"z",

		" ",
	}

	return NewVocabulary(tokens)
}
func NewVocabularyFromTexts(texts []string) (*Vocabulary, error) {
	if len(texts) == 0 {
		return nil, ErrEmptyVocabulary
	}

	seen := make(map[string]struct{})

	tokens := make([]string, 0)

	// CTC blank must have a stable ID.
	tokens = append(tokens, BlankToken)
	seen[BlankToken] = struct{}{}

	for _, text := range texts {
		text = strings.ToLower(strings.TrimSpace(text))

		if text == "" {
			continue
		}

		for _, r := range text {
			token := string(r)

			if _, exists := seen[token]; exists {
				continue
			}

			seen[token] = struct{}{}
			tokens = append(tokens, token)
		}
	}

	if len(tokens) <= 1 {
		return nil, ErrEmptyVocabulary
	}

	return NewVocabulary(tokens)
}
func NewTokenizer(vocabulary *Vocabulary) (*Tokenizer, error) {
	if vocabulary == nil {
		return nil, ErrEmptyVocabulary
	}

	return &Tokenizer{
		vocabulary: vocabulary,
	}, nil
}
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if t == nil || t.vocabulary == nil {
		return nil, ErrEmptyVocabulary
	}

	text = strings.ToLower(
		strings.TrimSpace(text),
	)

	if text == "" {
		return nil, ErrInvalidToken
	}

	result := make(
		[]int,
		0,
		utf8.RuneCountInString(text),
	)

	for _, r := range text {
		id, err := t.vocabulary.EncodeToken(
			string(r),
		)

		if err != nil {
			return nil, err
		}

		result = append(result, id)
	}

	return result, nil
}
func (t *Tokenizer) Decode(ids []int) (string, error) {
	if t == nil || t.vocabulary == nil {
		return "", ErrEmptyVocabulary
	}

	var builder strings.Builder

	for _, id := range ids {
		if id == t.vocabulary.BlankID() {
			continue
		}

		token, err := t.vocabulary.DecodeID(id)
		if err != nil {
			return "", err
		}

		builder.WriteString(token)
	}

	return builder.String(), nil
}

// Dataset
func NewDataset() *Dataset {
	return &Dataset{
		Samples: make([]TrainingSample, 0),
	}
}
func (d *Dataset) Add(sample TrainingSample) error {
	if d == nil {
		return ErrInvalidDataset
	}

	if sample.Features == nil {
		return ErrInvalidDataset
	}

	if len(sample.Labels) == 0 {
		return ErrInvalidDataset
	}

	if sample.Text == "" {
		return ErrInvalidDataset
	}

	d.Samples = append(d.Samples, sample)

	return nil
}
func (d *Dataset) Len() int {
	if d == nil {
		return 0
	}

	return len(d.Samples)
}
func (d *Dataset) Get(index int) (*TrainingSample, error) {
	if d == nil {
		return nil, ErrInvalidDataset
	}

	if index < 0 || index >= len(d.Samples) {
		return nil, ErrEmptyDataset
	}

	return &d.Samples[index], nil
}

// Dataset
// DatasetLoader
func NewDatasetLoader(
	tokenizer *Tokenizer,
	config FeatureConfig,
) (*DatasetLoader, error) {
	if tokenizer == nil {
		return nil, ErrInvalidDatasetLoader
	}

	if err := config.Validate(); err != nil {
		return nil, err
	}

	return &DatasetLoader{
		tokenizer: tokenizer,
		config:    config,
	}, nil
}
func (l *DatasetLoader) Load(
	manifestPath string,
) (*Dataset, error) {
	if l == nil || l.tokenizer == nil {
		return nil, ErrInvalidDatasetLoader
	}

	entries, err := LoadManifest(manifestPath)
	if err != nil {
		return nil, err
	}

	dataset := NewDataset()

	baseDir := filepath.Dir(manifestPath)

	for _, entry := range entries {
		audioPath := entry.AudioPath

		if !filepath.IsAbs(audioPath) {
			audioPath = filepath.Join(baseDir, audioPath)
		}

		audio, err := LoadWAV(audioPath)
		if err != nil {
			return nil, err
		}

		audio, err = NormalizeAudio(
			audio,
			l.config.SampleRate,
		)
		if err != nil {
			return nil, err
		}

		features, err := ExtractFeatures(
			audio.Samples,
			l.config,
		)
		if err != nil {
			return nil, err
		}

		labels, err := l.tokenizer.Encode(entry.Text)
		if err != nil {
			return nil, err
		}

		err = dataset.Add(TrainingSample{
			Features: features,
			Labels:   labels,
			Text:     entry.Text,
		})
		if err != nil {
			return nil, err
		}
	}

	if dataset.Len() == 0 {
		return nil, ErrEmptyDataset
	}

	return dataset, nil
}

// DatasetLoader
func (b *Batch) ToTensor() (*Tensor3D, error) {
	if b == nil || b.Features == nil {
		return nil, ErrInvalidBatch
	}

	tensor := NewTensor3D(
		b.BatchSize,
		b.TimeSteps,
		b.FeatureDim,
	)

	if tensor == nil {
		return nil, ErrInvalidTensor
	}

	for batch := 0; batch < b.BatchSize; batch++ {
		inputLength := b.InputLengths[batch]

		for time := 0; time < inputLength; time++ {
			for feature := 0; feature < b.FeatureDim; feature++ {
				value := b.Features.Get(
					batch*b.TimeSteps+time,
					feature,
				)

				if err := tensor.Set(
					batch,
					time,
					feature,
					value,
				); err != nil {
					return nil, err
				}
			}
		}
	}

	return tensor, nil
}
func NewTensor3D(
	batch int,
	time int,
	features int,
) *Tensor3D {
	if batch <= 0 || time <= 0 || features <= 0 {
		return nil
	}

	return &Tensor3D{
		Data:     make([]float32, batch*time*features),
		Batch:    batch,
		Time:     time,
		Features: features,
	}
}
func (t *Tensor3D) index(
	batch int,
	time int,
	feature int,
) int {
	return (batch*t.Time*t.Features + time*t.Features + feature)
}
func (t *Tensor3D) Set(
	batch int,
	time int,
	feature int,
	value float32,
) error {
	if t == nil {
		return ErrInvalidTensor
	}

	if batch < 0 || batch >= t.Batch ||
		time < 0 || time >= t.Time ||
		feature < 0 || feature >= t.Features {
		return ErrTensorIndex
	}

	t.Data[t.index(batch, time, feature)] = value

	return nil
}
func (t *Tensor3D) Get(
	batch int,
	time int,
	feature int,
) (float32, error) {
	if t == nil {
		return 0, ErrInvalidTensor
	}

	if batch < 0 || batch >= t.Batch ||
		time < 0 || time >= t.Time ||
		feature < 0 || feature >= t.Features {
		return 0, ErrTensorIndex
	}

	return t.Data[t.index(batch, time, feature)], nil
}
func batchToTensor3D(batch *Batch) *Tensor3D {
	if batch == nil ||
		batch.Features == nil ||
		batch.BatchSize <= 0 ||
		batch.TimeSteps <= 0 ||
		batch.FeatureDim <= 0 {
		return nil
	}

	tensor := NewTensor3D(
		batch.BatchSize,
		batch.TimeSteps,
		batch.FeatureDim,
	)

	for b := 0; b < batch.BatchSize; b++ {
		for t := 0; t < batch.TimeSteps; t++ {
			for f := 0; f < batch.FeatureDim; f++ {

				row := b*batch.TimeSteps + t

				value := batch.Features.Get(
					row,
					f,
				)

				if err := tensor.Set(
					b,
					t,
					f,
					value,
				); err != nil {
					return nil
				}
			}
		}
	}

	return tensor
}
func NewLinear(inFeatures, outFeatures int) *Linear {
	if inFeatures <= 0 || outFeatures <= 0 {
		return nil
	}

	return &Linear{
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
		Weights:     make([]float32, inFeatures*outFeatures),
		Bias:        make([]float32, outFeatures),
	}
}
func (l *Linear) weightIndex(out, in int) int {
	return out*l.InFeatures + in
}
func (l *Linear) SetWeight(out, in int, value float32) error {
	if l == nil {
		return ErrInvalidLinear
	}

	if out < 0 || out >= l.OutFeatures ||
		in < 0 || in >= l.InFeatures {
		return ErrLinearInput
	}

	l.Weights[l.weightIndex(out, in)] = value

	return nil
}
func (l *Linear) SetBias(out int, value float32) error {
	if l == nil {
		return ErrInvalidLinear
	}

	if out < 0 || out >= l.OutFeatures {
		return ErrLinearInput
	}

	l.Bias[out] = value

	return nil
}
func (l *Linear) Forward(input []float32) ([]float32, error) {
	if l == nil {
		return nil, ErrInvalidLinear
	}

	if len(input) != l.InFeatures {
		return nil, ErrLinearInput
	}

	output := make([]float32, l.OutFeatures)

	for out := 0; out < l.OutFeatures; out++ {
		sum := l.Bias[out]

		for in := 0; in < l.InFeatures; in++ {
			sum += l.Weights[l.weightIndex(out, in)] * input[in]
		}

		output[out] = sum
	}

	return output, nil
}
func NewLSTMCell(inputSize, hiddenSize int) *LSTMCell {
	if inputSize <= 0 || hiddenSize <= 0 {
		return nil
	}

	inputHidden := inputSize + hiddenSize

	return &LSTMCell{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Weights:    make([]float32, 4*hiddenSize*inputHidden),
		Bias:       make([]float32, 4*hiddenSize),
	}
}
func sigmoid(x float32) float32 {
	if x >= 0 {
		z := float32(math.Exp(float64(-x)))
		return 1 / (1 + z)
	}

	z := float32(math.Exp(float64(x)))
	return z / (1 + z)
}
func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}
func (l *LSTMCell) weightIndex(gate, hidden, input int) int {
	inputHidden := l.InputSize + l.HiddenSize

	return gate*l.HiddenSize*inputHidden +
		hidden*inputHidden +
		input
}
func (l *LSTMCell) SetWeight(
	gate int,
	hidden int,
	input int,
	value float32,
) error {
	if l == nil {
		return ErrInvalidLSTM
	}

	if gate < 0 || gate >= 4 ||
		hidden < 0 || hidden >= l.HiddenSize ||
		input < 0 || input >= l.InputSize+l.HiddenSize {
		return ErrLSTMInput
	}

	l.Weights[l.weightIndex(gate, hidden, input)] = value

	return nil
}
func (l *LSTMCell) SetBias(
	gate int,
	hidden int,
	value float32,
) error {
	if l == nil {
		return ErrInvalidLSTM
	}

	if gate < 0 || gate >= 4 ||
		hidden < 0 || hidden >= l.HiddenSize {
		return ErrLSTMInput
	}

	l.Bias[gate*l.HiddenSize+hidden] = value

	return nil
}
func (l *LSTMCell) Forward(
	input []float32,
	hidden []float32,
	cell []float32,
) ([]float32, []float32, error) {

	if l == nil {
		return nil, nil, ErrInvalidLSTM
	}

	if len(input) != l.InputSize ||
		len(hidden) != l.HiddenSize ||
		len(cell) != l.HiddenSize {
		return nil, nil, ErrLSTMInput
	}

	inputHidden := l.InputSize + l.HiddenSize

	combined := make([]float32, inputHidden)

	copy(combined, input)
	copy(combined[l.InputSize:], hidden)

	gates := make([]float32, 4*l.HiddenSize)

	for gate := 0; gate < 4; gate++ {
		for h := 0; h < l.HiddenSize; h++ {
			sum := l.Bias[gate*l.HiddenSize+h]

			for i := 0; i < inputHidden; i++ {
				sum += l.Weights[l.weightIndex(gate, h, i)] * combined[i]
			}

			gates[gate*l.HiddenSize+h] = sum
		}
	}

	nextHidden := make([]float32, l.HiddenSize)
	nextCell := make([]float32, l.HiddenSize)

	for h := 0; h < l.HiddenSize; h++ {
		// Gate order:
		// input, forget, candidate, output

		i := sigmoid(gates[h])

		f := sigmoid(
			gates[l.HiddenSize+h],
		)

		g := tanh(
			gates[2*l.HiddenSize+h],
		)

		o := sigmoid(
			gates[3*l.HiddenSize+h],
		)

		nextCell[h] =
			f*cell[h] +
				i*g

		nextHidden[h] =
			o * tanh(nextCell[h])
	}

	return nextHidden, nextCell, nil
}
func NewLSTMSequence(
	inputSize int,
	hiddenSize int,
) *LSTMSequence {
	cell := NewLSTMCell(inputSize, hiddenSize)

	if cell == nil {
		return nil
	}

	return &LSTMSequence{
		Cell: cell,
	}
}
func (l *LSTMSequence) Forward(
	input *Tensor3D,
	inputLengths []int,
) (*Tensor3D, error) {

	if l == nil || l.Cell == nil {
		return nil, ErrInvalidLSTMSequence
	}

	if input == nil {
		return nil, ErrInvalidLSTMSequence
	}

	if input.Features != l.Cell.InputSize {
		return nil, ErrInvalidLSTMSequence
	}

	if len(inputLengths) != input.Batch {
		return nil, ErrInvalidLSTMSequence
	}

	output := NewTensor3D(
		input.Batch,
		input.Time,
		l.Cell.HiddenSize,
	)

	if output == nil {
		return nil, ErrInvalidLSTMSequence
	}

	for batch := 0; batch < input.Batch; batch++ {
		length := inputLengths[batch]

		if length < 0 || length > input.Time {
			return nil, ErrInvalidLSTMSequence
		}

		hidden := make([]float32, l.Cell.HiddenSize)
		cell := make([]float32, l.Cell.HiddenSize)

		for time := 0; time < length; time++ {
			x := make([]float32, l.Cell.InputSize)

			for feature := 0; feature < l.Cell.InputSize; feature++ {
				value, err := input.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				x[feature] = value
			}

			nextHidden, nextCell, err :=
				l.Cell.Forward(
					x,
					hidden,
					cell,
				)

			if err != nil {
				return nil, err
			}

			hidden = nextHidden
			cell = nextCell

			for feature := 0; feature < l.Cell.HiddenSize; feature++ {
				if err := output.Set(
					batch,
					time,
					feature,
					hidden[feature],
				); err != nil {
					return nil, err
				}
			}
		}
	}

	return output, nil
}
func NewBackwardLSTMSequence(
	inputSize int,
	hiddenSize int,
) *BackwardLSTMSequence {
	cell := NewLSTMCell(inputSize, hiddenSize)

	if cell == nil {
		return nil
	}

	return &BackwardLSTMSequence{
		Cell: cell,
	}
}
func (l *BackwardLSTMSequence) Forward(
	input *Tensor3D,
	inputLengths []int,
) (*Tensor3D, error) {

	if l == nil || l.Cell == nil {
		return nil, ErrInvalidBackwardLSTM
	}

	if input == nil {
		return nil, ErrInvalidBackwardLSTM
	}

	if input.Features != l.Cell.InputSize {
		return nil, ErrInvalidBackwardLSTM
	}

	if len(inputLengths) != input.Batch {
		return nil, ErrInvalidBackwardLSTM
	}

	output := NewTensor3D(
		input.Batch,
		input.Time,
		l.Cell.HiddenSize,
	)

	if output == nil {
		return nil, ErrInvalidBackwardLSTM
	}

	for batch := 0; batch < input.Batch; batch++ {
		length := inputLengths[batch]

		if length < 0 || length > input.Time {
			return nil, ErrInvalidBackwardLSTM
		}

		hidden := make([]float32, l.Cell.HiddenSize)
		cell := make([]float32, l.Cell.HiddenSize)

		for time := length - 1; time >= 0; time-- {
			x := make([]float32, l.Cell.InputSize)

			for feature := 0; feature < l.Cell.InputSize; feature++ {
				value, err := input.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				x[feature] = value
			}

			nextHidden, nextCell, err :=
				l.Cell.Forward(
					x,
					hidden,
					cell,
				)

			if err != nil {
				return nil, err
			}

			hidden = nextHidden
			cell = nextCell

			for feature := 0; feature < l.Cell.HiddenSize; feature++ {
				if err := output.Set(
					batch,
					time,
					feature,
					hidden[feature],
				); err != nil {
					return nil, err
				}
			}
		}
	}

	return output, nil
}
func NewBiLSTM(
	inputSize int,
	hiddenSize int,
) *BiLSTM {
	if inputSize <= 0 || hiddenSize <= 0 {
		return nil
	}

	forward := NewLSTMSequence(
		inputSize,
		hiddenSize,
	)

	backward := NewBackwardLSTMSequence(
		inputSize,
		hiddenSize,
	)

	if forward == nil || backward == nil {
		return nil
	}

	return &BiLSTM{
		ForwardLSTM:  forward,
		BackwardLSTM: backward,
		InputSize:    inputSize,
		HiddenSize:   hiddenSize,
	}
}
func (b *BiLSTM) Forward(
	input *Tensor3D,
	inputLengths []int,
) (*Tensor3D, error) {

	if b == nil ||
		b.ForwardLSTM == nil ||
		b.BackwardLSTM == nil {
		return nil, ErrInvalidBiLSTM
	}

	if input == nil {
		return nil, ErrInvalidBiLSTM
	}

	if input.Features != b.InputSize {
		return nil, ErrInvalidBiLSTM
	}

	forwardOutput, err := b.ForwardLSTM.Forward(
		input,
		inputLengths,
	)

	if err != nil {
		return nil, err
	}

	backwardOutput, err := b.BackwardLSTM.Forward(
		input,
		inputLengths,
	)

	if err != nil {
		return nil, err
	}

	output := NewTensor3D(
		input.Batch,
		input.Time,
		b.HiddenSize*2,
	)

	if output == nil {
		return nil, ErrInvalidBiLSTM
	}

	for batch := 0; batch < input.Batch; batch++ {
		for time := 0; time < input.Time; time++ {

			// Forward output.
			for feature := 0; feature < b.HiddenSize; feature++ {
				value, err := forwardOutput.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				if err := output.Set(
					batch,
					time,
					feature,
					value,
				); err != nil {
					return nil, err
				}
			}

			// Backward output.
			for feature := 0; feature < b.HiddenSize; feature++ {
				value, err := backwardOutput.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				if err := output.Set(
					batch,
					time,
					b.HiddenSize+feature,
					value,
				); err != nil {
					return nil, err
				}
			}
		}
	}

	return output, nil
}
func (l *Linear) ForwardTensor(
	input *Tensor3D,
) (*Tensor3D, error) {

	if l == nil {
		return nil, ErrInvalidLinear
	}

	if input == nil {
		return nil, ErrLinearInput
	}

	if input.Features != l.InFeatures {
		return nil, ErrLinearInput
	}

	output := NewTensor3D(
		input.Batch,
		input.Time,
		l.OutFeatures,
	)

	if output == nil {
		return nil, ErrInvalidLinear
	}

	for batch := 0; batch < input.Batch; batch++ {
		for time := 0; time < input.Time; time++ {

			x := make([]float32, l.InFeatures)

			for feature := 0; feature < l.InFeatures; feature++ {
				value, err := input.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				x[feature] = value
			}

			y, err := l.Forward(x)
			if err != nil {
				return nil, err
			}

			for feature := 0; feature < l.OutFeatures; feature++ {
				if err := output.Set(
					batch,
					time,
					feature,
					y[feature],
				); err != nil {
					return nil, err
				}
			}
		}
	}

	return output, nil
}
func NewCTCDecoder(
	vocabulary *Vocabulary,
) *CTCDecoder {
	if vocabulary == nil {
		return nil
	}

	return &CTCDecoder{
		Vocabulary: vocabulary,
		BlankID:    vocabulary.BlankID(),
	}
}
func (d *CTCDecoder) Decode(
	logits *Tensor3D,
	inputLengths []int,
) ([]string, error) {

	if d == nil || d.Vocabulary == nil {
		return nil, ErrInvalidCTCDecoder
	}

	if logits == nil {
		return nil, ErrInvalidCTCDecoder
	}

	if len(inputLengths) != logits.Batch {
		return nil, ErrInvalidCTCDecoder
	}

	if logits.Features != d.Vocabulary.Size() {
		return nil, ErrInvalidCTCDecoder
	}

	results := make([]string, logits.Batch)

	for batch := 0; batch < logits.Batch; batch++ {
		length := inputLengths[batch]

		if length < 0 || length > logits.Time {
			return nil, ErrInvalidCTCDecoder
		}

		ids := make([]int, 0, length)

		previous := -1

		for time := 0; time < length; time++ {
			bestID := 0
			bestValue := float32(-1e30)

			for token := 0; token < logits.Features; token++ {
				value, err := logits.Get(
					batch,
					time,
					token,
				)

				if err != nil {
					return nil, err
				}

				if value > bestValue {
					bestValue = value
					bestID = token
				}
			}

			// Collapse only consecutive repeats.
			if bestID == previous {
				continue
			}

			previous = bestID

			// Blank is removed after repeat collapsing.
			if bestID == d.BlankID {
				continue
			}

			ids = append(ids, bestID)
		}

		text, err := d.VocabularyDecode(ids)
		if err != nil {
			return nil, err
		}

		results[batch] = text
	}

	return results, nil
}
func (d *CTCDecoder) VocabularyDecode(
	ids []int,
) (string, error) {
	if d == nil || d.Vocabulary == nil {
		return "", ErrInvalidCTCDecoder
	}

	tokenizer, err := NewTokenizer(d.Vocabulary)
	if err != nil {
		return "", err
	}

	return tokenizer.Decode(ids)
}
func NewSTTModel(
	inputSize int,
	hiddenSize int,
	vocabulary *Vocabulary,
) *STTModel {

	if inputSize <= 0 ||
		hiddenSize <= 0 ||
		vocabulary == nil {
		return nil
	}

	encoder := NewBiLSTM(
		inputSize,
		hiddenSize,
	)

	if encoder == nil {
		return nil
	}

	decoder := NewLinear(
		hiddenSize*2,
		vocabulary.Size(),
	)

	if decoder == nil {
		return nil
	}

	rng := rand.New(
		rand.NewSource(42),
	)

	InitializeLinear(
		decoder,
		rng,
	)

	InitializeLSTMCell(
		encoder.ForwardLSTM.Cell,
		rng,
	)

	InitializeLSTMCell(
		encoder.BackwardLSTM.Cell,
		rng,
	)

	ctc := NewCTCDecoder(
		vocabulary,
	)

	if ctc == nil {
		return nil
	}

	return &STTModel{
		Encoder: encoder,
		Decoder: decoder,
		CTC:     ctc,

		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		OutputSize: vocabulary.Size(),
	}
}
func (m *STTModel) Forward(
	input *Tensor3D,
	inputLengths []int,
) (*Tensor3D, error) {

	if m == nil ||
		m.Encoder == nil ||
		m.Decoder == nil {
		return nil, ErrInvalidSTTModel
	}

	if input == nil {
		return nil, ErrInvalidSTTModel
	}

	if input.Features != m.InputSize {
		return nil, ErrInvalidSTTModel
	}

	encoded, err := m.Encoder.Forward(
		input,
		inputLengths,
	)

	if err != nil {
		return nil, err
	}

	logits, err := m.Decoder.ForwardTensor(
		encoded,
	)

	if err != nil {
		return nil, err
	}

	return logits, nil
}
func (m *STTModel) Transcribe(
	input *Tensor3D,
	inputLengths []int,
) ([]string, error) {

	if m == nil || m.CTC == nil {
		return nil, ErrInvalidSTTModel
	}

	logits, err := m.Forward(
		input,
		inputLengths,
	)

	if err != nil {
		return nil, err
	}

	return m.CTC.Decode(
		logits,
		inputLengths,
	)
}
func (l *Linear) Backward(
	input []float32,
	gradient []float32,
) ([]float32, *LinearGradient, error) {

	if l == nil {
		return nil, nil, ErrInvalidLinearBackward
	}

	if len(input) != l.InFeatures ||
		len(gradient) != l.OutFeatures {
		return nil, nil, ErrInvalidLinearBackward
	}

	inputGradient := make([]float32, l.InFeatures)

	weightGradient := make(
		[]float32,
		len(l.Weights),
	)

	biasGradient := make(
		[]float32,
		len(l.Bias),
	)

	for out := 0; out < l.OutFeatures; out++ {
		dY := gradient[out]

		biasGradient[out] = dY

		for in := 0; in < l.InFeatures; in++ {
			index := l.weightIndex(out, in)

			weightGradient[index] +=
				dY * input[in]

			inputGradient[in] +=
				dY * l.Weights[index]
		}
	}

	return inputGradient, &LinearGradient{
		Weights: weightGradient,
		Bias:    biasGradient,
	}, nil
}
func (l *Linear) BackwardTensor(
	input *Tensor3D,
	gradient *Tensor3D,
) (*Tensor3D, *LinearGradient, error) {

	if l == nil {
		return nil, nil, ErrInvalidLinearBackward
	}

	if input == nil || gradient == nil {
		return nil, nil, ErrInvalidLinearBackward
	}

	if input.Batch != gradient.Batch ||
		input.Time != gradient.Time {
		return nil, nil, ErrInvalidLinearBackward
	}

	if input.Features != l.InFeatures ||
		gradient.Features != l.OutFeatures {
		return nil, nil, ErrInvalidLinearBackward
	}

	inputGradient := NewTensor3D(
		input.Batch,
		input.Time,
		l.InFeatures,
	)

	if inputGradient == nil {
		return nil, nil, ErrInvalidLinearBackward
	}

	weightGradient := make(
		[]float32,
		len(l.Weights),
	)

	biasGradient := make(
		[]float32,
		len(l.Bias),
	)

	for batch := 0; batch < input.Batch; batch++ {
		for time := 0; time < input.Time; time++ {

			for out := 0; out < l.OutFeatures; out++ {

				dY, err := gradient.Get(
					batch,
					time,
					out,
				)

				if err != nil {
					return nil, nil, err
				}

				biasGradient[out] += dY

				for in := 0; in < l.InFeatures; in++ {

					x, err := input.Get(
						batch,
						time,
						in,
					)

					if err != nil {
						return nil, nil, err
					}

					index := l.weightIndex(
						out,
						in,
					)

					weightGradient[index] +=
						dY * x

					inputValue, err :=
						inputGradient.Get(
							batch,
							time,
							in,
						)

					if err != nil {
						return nil, nil, err
					}

					inputValue +=
						dY * l.Weights[index]

					if err := inputGradient.Set(
						batch,
						time,
						in,
						inputValue,
					); err != nil {
						return nil, nil, err
					}
				}
			}
		}
	}

	return inputGradient, &LinearGradient{
		Weights: weightGradient,
		Bias:    biasGradient,
	}, nil
}
func (l *LSTMCell) Backward(
	input []float32,
	hidden []float32,
	cell []float32,

	nextHiddenGradient []float32,
	nextCellGradient []float32,
) (*LSTMBackwardResult, error) {

	if l == nil {
		return nil, ErrInvalidLSTMBackward
	}

	if len(input) != l.InputSize ||
		len(hidden) != l.HiddenSize ||
		len(cell) != l.HiddenSize ||
		len(nextHiddenGradient) != l.HiddenSize ||
		len(nextCellGradient) != l.HiddenSize {
		return nil, ErrInvalidLSTMBackward
	}

	inputHidden := l.InputSize + l.HiddenSize

	combined := make([]float32, inputHidden)

	copy(combined, input)
	copy(combined[l.InputSize:], hidden)

	// Forward gate values.
	gates := make([]float32, 4*l.HiddenSize)

	for gate := 0; gate < 4; gate++ {
		for h := 0; h < l.HiddenSize; h++ {

			sum := l.Bias[gate*l.HiddenSize+h]

			for i := 0; i < inputHidden; i++ {
				sum += l.Weights[l.weightIndex(gate, h, i)] * combined[i]
			}

			gates[gate*l.HiddenSize+h] = sum
		}
	}

	inputGate := make([]float32, l.HiddenSize)
	forgetGate := make([]float32, l.HiddenSize)
	candidateGate := make([]float32, l.HiddenSize)
	outputGate := make([]float32, l.HiddenSize)

	for h := 0; h < l.HiddenSize; h++ {
		inputGate[h] = sigmoid(
			gates[h],
		)

		forgetGate[h] = sigmoid(
			gates[l.HiddenSize+h],
		)

		candidateGate[h] = tanh(
			gates[2*l.HiddenSize+h],
		)

		outputGate[h] = sigmoid(
			gates[3*l.HiddenSize+h],
		)
	}

	nextCell := make([]float32, l.HiddenSize)

	for h := 0; h < l.HiddenSize; h++ {
		nextCell[h] =
			forgetGate[h]*cell[h] +
				inputGate[h]*candidateGate[h]
	}

	inputGradient := make([]float32, l.InputSize)
	hiddenGradient := make([]float32, l.HiddenSize)
	cellGradient := make([]float32, l.HiddenSize)

	weightGradient := make(
		[]float32,
		len(l.Weights),
	)

	biasGradient := make(
		[]float32,
		len(l.Bias),
	)

	// ------------------------------------------------------------
	// Backpropagate through h_t = o_t * tanh(c_t)
	// ------------------------------------------------------------

	for h := 0; h < l.HiddenSize; h++ {

		tanhCell := tanh(nextCell[h])

		// dh/dc
		dCell :=
			nextHiddenGradient[h] *
				outputGate[h] *
				(1 - tanhCell*tanhCell)

		// Gradient coming from c_{t+1}.
		dCell += nextCellGradient[h]

		// --------------------------------------------------------
		// Cell equation:
		//
		// c_t = f*c_prev + i*g
		// --------------------------------------------------------

		dForget :=
			dCell * cell[h]

		dInput :=
			dCell * candidateGate[h]

		dCandidate :=
			dCell * inputGate[h]

		cellGradient[h] =
			dCell * forgetGate[h]

		// --------------------------------------------------------
		// Gate activation derivatives.
		// --------------------------------------------------------

		dForgetPre :=
			dForget *
				forgetGate[h] *
				(1 - forgetGate[h])

		dInputPre :=
			dInput *
				inputGate[h] *
				(1 - inputGate[h])

		dCandidatePre :=
			dCandidate *
				(1 - candidateGate[h]*
					candidateGate[h])

		dOutputPre :=
			nextHiddenGradient[h] *
				tanhCell *
				outputGate[h] *
				(1 - outputGate[h])

		gateGradients := []float32{
			dInputPre,
			dForgetPre,
			dCandidatePre,
			dOutputPre,
		}

		// --------------------------------------------------------
		// Parameter gradients + gradient to combined input.
		// --------------------------------------------------------

		for gate := 0; gate < 4; gate++ {

			dGate :=
				gateGradients[gate]

			biasGradient[gate*l.HiddenSize+h] += dGate

			for i := 0; i < inputHidden; i++ {

				index := l.weightIndex(
					gate,
					h,
					i,
				)

				weightGradient[index] +=
					dGate * combined[i]

				if i < l.InputSize {
					inputGradient[i] +=
						dGate *
							l.Weights[index]
				} else {
					hiddenGradient[i-l.InputSize] +=
						dGate *
							l.Weights[index]
				}
			}
		}
	}

	return &LSTMBackwardResult{
		InputGradient:  inputGradient,
		HiddenGradient: hiddenGradient,
		CellGradient:   cellGradient,
		Parameters: &LSTMCellGradient{
			Weights: weightGradient,
			Bias:    biasGradient,
		},
	}, nil
}
func (l *LSTMSequence) Backward(
	input *Tensor3D,
	inputLengths []int,
	outputGradient *Tensor3D,
) (*LSTMSequenceGradient, error) {

	if l == nil || l.Cell == nil {
		return nil, ErrInvalidLSTMSequenceBackward
	}

	if input == nil || outputGradient == nil {
		return nil, ErrInvalidLSTMSequenceBackward
	}

	if input.Batch != outputGradient.Batch ||
		input.Time != outputGradient.Time {
		return nil, ErrInvalidLSTMSequenceBackward
	}

	if input.Features != l.Cell.InputSize ||
		outputGradient.Features != l.Cell.HiddenSize {
		return nil, ErrInvalidLSTMSequenceBackward
	}

	if len(inputLengths) != input.Batch {
		return nil, ErrInvalidLSTMSequenceBackward
	}

	inputGradient := NewTensor3D(
		input.Batch,
		input.Time,
		l.Cell.InputSize,
	)

	if inputGradient == nil {
		return nil, ErrInvalidLSTMSequenceBackward
	}

	totalWeightGradient := make(
		[]float32,
		len(l.Cell.Weights),
	)

	totalBiasGradient := make(
		[]float32,
		len(l.Cell.Bias),
	)

	for batch := 0; batch < input.Batch; batch++ {

		length := inputLengths[batch]

		if length < 0 || length > input.Time {
			return nil, ErrInvalidLSTMSequenceBackward
		}

		// --------------------------------------------------------
		// Save forward states.
		// --------------------------------------------------------

		hiddens := make([][]float32, length+1)
		cells := make([][]float32, length+1)

		hiddens[0] = make([]float32, l.Cell.HiddenSize)
		cells[0] = make([]float32, l.Cell.HiddenSize)

		for time := 0; time < length; time++ {

			x := make([]float32, l.Cell.InputSize)

			for feature := 0; feature < l.Cell.InputSize; feature++ {
				value, err := input.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				x[feature] = value
			}

			nextHidden, nextCell, err :=
				l.Cell.Forward(
					x,
					hiddens[time],
					cells[time],
				)

			if err != nil {
				return nil, err
			}

			hiddens[time+1] = nextHidden
			cells[time+1] = nextCell
		}

		// --------------------------------------------------------
		// Gradient flowing from the future timestep.
		// --------------------------------------------------------

		hiddenGradient := make(
			[]float32,
			l.Cell.HiddenSize,
		)

		cellGradient := make(
			[]float32,
			l.Cell.HiddenSize,
		)

		// --------------------------------------------------------
		// Backward through time.
		// --------------------------------------------------------

		for time := length - 1; time >= 0; time-- {

			x := make([]float32, l.Cell.InputSize)

			for feature := 0; feature < l.Cell.InputSize; feature++ {
				value, err := input.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				x[feature] = value
			}

			outputGrad := make(
				[]float32,
				l.Cell.HiddenSize,
			)

			for feature := 0; feature < l.Cell.HiddenSize; feature++ {
				value, err := outputGradient.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				outputGrad[feature] = value
			}

			// Add gradient coming from future timestep.
			for feature := 0; feature < l.Cell.HiddenSize; feature++ {
				outputGrad[feature] += hiddenGradient[feature]
			}

			result, err := l.Cell.Backward(
				x,
				hiddens[time],
				cells[time],
				outputGrad,
				cellGradient,
			)

			if err != nil {
				return nil, err
			}

			for feature := 0; feature < l.Cell.InputSize; feature++ {
				if err := inputGradient.Set(
					batch,
					time,
					feature,
					result.InputGradient[feature],
				); err != nil {
					return nil, err
				}
			}

			hiddenGradient =
				result.HiddenGradient

			cellGradient =
				result.CellGradient

			for i := range totalWeightGradient {
				totalWeightGradient[i] +=
					result.Parameters.Weights[i]
			}

			for i := range totalBiasGradient {
				totalBiasGradient[i] +=
					result.Parameters.Bias[i]
			}
		}
	}

	return &LSTMSequenceGradient{
		Input: inputGradient,
		Parameters: &LSTMCellGradient{
			Weights: totalWeightGradient,
			Bias:    totalBiasGradient,
		},
	}, nil
}
func (l *BackwardLSTMSequence) Backward(
	input *Tensor3D,
	inputLengths []int,
	outputGradient *Tensor3D,
) (*LSTMSequenceGradient, error) {

	if l == nil || l.Cell == nil {
		return nil, ErrInvalidBackwardLSTMSequenceBackward
	}

	if input == nil || outputGradient == nil {
		return nil, ErrInvalidBackwardLSTMSequenceBackward
	}

	if input.Batch != outputGradient.Batch ||
		input.Time != outputGradient.Time {
		return nil, ErrInvalidBackwardLSTMSequenceBackward
	}

	if input.Features != l.Cell.InputSize ||
		outputGradient.Features != l.Cell.HiddenSize {
		return nil, ErrInvalidBackwardLSTMSequenceBackward
	}

	if len(inputLengths) != input.Batch {
		return nil, ErrInvalidBackwardLSTMSequenceBackward
	}

	for _, length := range inputLengths {
		if length < 0 || length > input.Time {
			return nil, ErrInvalidBackwardLSTMSequenceBackward
		}
	}

	inputGradient := NewTensor3D(
		input.Batch,
		input.Time,
		input.Features,
	)

	if inputGradient == nil {
		return nil, ErrInvalidBackwardLSTMSequenceBackward
	}

	parameters := &LSTMCellGradient{
		Weights: make([]float32, len(l.Cell.Weights)),
		Bias:    make([]float32, len(l.Cell.Bias)),
	}

	for batch := 0; batch < input.Batch; batch++ {

		length := inputLengths[batch]

		if length == 0 {
			continue
		}

		/*
			BackwardLSTM forward order:

			    original time:
			    0 1 2 3

			    processing:
			    3 -> 2 -> 1 -> 0

			For every original timestep we store the state
			AFTER that timestep was processed.

			hiddens[t] = hidden produced at original time t
			cells[t]   = cell produced at original time t
		*/

		hiddens := make([][]float32, length)
		cells := make([][]float32, length)

		for time := 0; time < length; time++ {
			hiddens[time] = make(
				[]float32,
				l.Cell.HiddenSize,
			)

			cells[time] = make(
				[]float32,
				l.Cell.HiddenSize,
			)
		}

		hidden := make(
			[]float32,
			l.Cell.HiddenSize,
		)

		cell := make(
			[]float32,
			l.Cell.HiddenSize,
		)

		/*
			Reconstruct forward states.

			3 -> 2 -> 1 -> 0
		*/
		for time := length - 1; time >= 0; time-- {

			x := make(
				[]float32,
				input.Features,
			)

			for feature := 0; feature < input.Features; feature++ {

				value, err := input.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				x[feature] = value
			}

			nextHidden, nextCell, err :=
				l.Cell.Forward(
					x,
					hidden,
					cell,
				)

			if err != nil {
				return nil, err
			}

			copy(hidden, nextHidden)
			copy(cell, nextCell)

			copy(hiddens[time], hidden)
			copy(cells[time], cell)
		}

		/*
			Because the forward processing order was:

			    length-1 -> ... -> 1 -> 0

			the backward pass must be:

			    0 -> 1 -> ... -> length-1
		*/

		nextHiddenGradient := make(
			[]float32,
			l.Cell.HiddenSize,
		)

		nextCellGradient := make(
			[]float32,
			l.Cell.HiddenSize,
		)

		for time := 0; time < length; time++ {

			x := make(
				[]float32,
				input.Features,
			)

			for feature := 0; feature < input.Features; feature++ {

				value, err := input.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				x[feature] = value
			}

			/*
				Find the state BEFORE this timestep was
				processed in the reversed direction.

				For original time t:

				    previous state = state at t+1

				Except for the first processed timestep,
				which is original time length-1 and starts
				from zero state.
			*/

			previousHidden := make(
				[]float32,
				l.Cell.HiddenSize,
			)

			previousCell := make(
				[]float32,
				l.Cell.HiddenSize,
			)

			if time < length-1 {
				copy(
					previousHidden,
					hiddens[time+1],
				)

				copy(
					previousCell,
					cells[time+1],
				)
			}

			/*
				Gradient coming from the output at this
				original timestep.
			*/
			gradient := make(
				[]float32,
				l.Cell.HiddenSize,
			)

			for feature := 0; feature < l.Cell.HiddenSize; feature++ {

				value, err := outputGradient.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				/*
					The cell receives:

					    output gradient
					    +
					    gradient from the next
					    recurrent timestep
				*/
				gradient[feature] =
					value +
						nextHiddenGradient[feature]
			}

			result, err := l.Cell.Backward(
				x,
				previousHidden,
				previousCell,
				gradient,
				nextCellGradient,
			)

			if err != nil {
				return nil, err
			}

			/*
				Store gradient with original time index.
			*/
			for feature := 0; feature < input.Features; feature++ {

				if err := inputGradient.Set(
					batch,
					time,
					feature,
					result.InputGradient[feature],
				); err != nil {
					return nil, err
				}
			}

			/*
				Accumulate parameter gradients.
			*/
			for i := range parameters.Weights {

				parameters.Weights[i] +=
					result.Parameters.Weights[i]
			}

			for i := range parameters.Bias {

				parameters.Bias[i] +=
					result.Parameters.Bias[i]
			}

			/*
				Pass recurrent gradients to the next
				backpropagation step.
			*/
			copy(
				nextHiddenGradient,
				result.HiddenGradient,
			)

			copy(
				nextCellGradient,
				result.CellGradient,
			)
		}
	}

	return &LSTMSequenceGradient{
		Input:      inputGradient,
		Parameters: parameters,
	}, nil
}
func (b *BiLSTM) Backward(
	input *Tensor3D,
	inputLengths []int,
	outputGradient *Tensor3D,
) (*BiLSTMGradient, error) {

	if b == nil ||
		b.ForwardLSTM == nil ||
		b.BackwardLSTM == nil {
		return nil, ErrInvalidBiLSTMBackward
	}

	if input == nil || outputGradient == nil {
		return nil, ErrInvalidBiLSTMBackward
	}

	if input.Batch != outputGradient.Batch ||
		input.Time != outputGradient.Time {
		return nil, ErrInvalidBiLSTMBackward
	}

	if input.Features != b.InputSize {
		return nil, ErrInvalidBiLSTMBackward
	}

	if outputGradient.Features != b.HiddenSize*2 {
		return nil, ErrInvalidBiLSTMBackward
	}

	if len(inputLengths) != input.Batch {
		return nil, ErrInvalidBiLSTMBackward
	}

	for _, length := range inputLengths {
		if length < 0 || length > input.Time {
			return nil, ErrInvalidBiLSTMBackward
		}
	}

	/*
		Split BiLSTM gradient into:

		    Forward:
		    [0 ... HiddenSize-1]

		    Backward:
		    [HiddenSize ... HiddenSize*2-1]
	*/

	forwardGradient := NewTensor3D(
		input.Batch,
		input.Time,
		b.HiddenSize,
	)

	backwardGradient := NewTensor3D(
		input.Batch,
		input.Time,
		b.HiddenSize,
	)

	if forwardGradient == nil ||
		backwardGradient == nil {
		return nil, ErrInvalidBiLSTMBackward
	}

	for batch := 0; batch < input.Batch; batch++ {
		for time := 0; time < input.Time; time++ {

			for feature := 0; feature < b.HiddenSize; feature++ {

				// Forward branch gradient.
				value, err := outputGradient.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				if err := forwardGradient.Set(
					batch,
					time,
					feature,
					value,
				); err != nil {
					return nil, err
				}

				// Backward branch gradient.
				value, err = outputGradient.Get(
					batch,
					time,
					b.HiddenSize+feature,
				)

				if err != nil {
					return nil, err
				}

				if err := backwardGradient.Set(
					batch,
					time,
					feature,
					value,
				); err != nil {
					return nil, err
				}
			}
		}
	}

	/*
		Backward through forward LSTM.
	*/

	forwardResult, err :=
		b.ForwardLSTM.Backward(
			input,
			inputLengths,
			forwardGradient,
		)

	if err != nil {
		return nil, err
	}

	if forwardResult == nil ||
		forwardResult.Input == nil ||
		forwardResult.Parameters == nil {
		return nil, ErrInvalidBiLSTMBackward
	}

	/*
		Backward through backward LSTM.

		BackwardLSTM has its own Backward implementation
		because its forward processing order is:

		    T-1 -> ... -> 1 -> 0
	*/

	backwardResult, err :=
		b.BackwardLSTM.Backward(
			input,
			inputLengths,
			backwardGradient,
		)

	if err != nil {
		return nil, err
	}

	if backwardResult == nil ||
		backwardResult.Input == nil ||
		backwardResult.Parameters == nil {
		return nil, ErrInvalidBiLSTMBackward
	}

	/*
		Combine input gradients.

		    dInput =
		        dInputForward +
		        dInputBackward
	*/

	inputGradient := NewTensor3D(
		input.Batch,
		input.Time,
		input.Features,
	)

	if inputGradient == nil {
		return nil, ErrInvalidBiLSTMBackward
	}

	for batch := 0; batch < input.Batch; batch++ {
		for time := 0; time < input.Time; time++ {
			for feature := 0; feature < input.Features; feature++ {

				forwardValue, err :=
					forwardResult.Input.Get(
						batch,
						time,
						feature,
					)

				if err != nil {
					return nil, err
				}

				backwardValue, err :=
					backwardResult.Input.Get(
						batch,
						time,
						feature,
					)

				if err != nil {
					return nil, err
				}

				if err := inputGradient.Set(
					batch,
					time,
					feature,
					forwardValue+backwardValue,
				); err != nil {
					return nil, err
				}
			}
		}
	}

	return &BiLSTMGradient{
		Input: inputGradient,

		ForwardParameters: forwardResult.Parameters,

		BackwardParameters: backwardResult.Parameters,
	}, nil
}
func (m *STTModel) Backward(
	input *Tensor3D,
	inputLengths []int,
	targets []int,
	targetLengths []int,
) (*STTModelGradient, error) {

	if m == nil ||
		m.Encoder == nil ||
		m.Decoder == nil ||
		m.CTC == nil {
		return nil, ErrInvalidSTTModelBackward
	}

	if input == nil {
		return nil, ErrInvalidSTTModelBackward
	}

	if input.Features != m.InputSize {
		return nil, ErrInvalidSTTModelBackward
	}

	if len(inputLengths) != input.Batch {
		return nil, ErrInvalidSTTModelBackward
	}

	if len(targetLengths) != input.Batch {
		return nil, ErrInvalidSTTModelBackward
	}

	totalTargets := 0

	for _, length := range inputLengths {
		if length <= 0 || length > input.Time {
			return nil, ErrInvalidSTTModelBackward
		}
	}

	for _, length := range targetLengths {
		if length < 0 {
			return nil, ErrInvalidSTTModelBackward
		}

		totalTargets += length
	}

	if totalTargets != len(targets) {
		return nil, ErrInvalidSTTModelBackward
	}

	/*
		Forward pass:

		    input
		      ↓
		    BiLSTM
		      ↓
		    Linear
		      ↓
		    logits
	*/

	encoderOutput, err := m.Encoder.Forward(
		input,
		inputLengths,
	)

	if err != nil {
		return nil, err
	}

	if encoderOutput == nil {
		return nil, ErrInvalidSTTModelBackward
	}

	logits, err := m.Decoder.ForwardTensor(
		encoderOutput,
	)

	if err != nil {
		return nil, err
	}

	if logits == nil {
		return nil, ErrInvalidSTTModelBackward
	}

	if logits.Batch != input.Batch ||
		logits.Time != input.Time ||
		logits.Features != m.OutputSize {
		return nil, ErrInvalidSTTModelBackward
	}

	/*
		CTC gradient.

		We need:

		    dLogits
	*/

	logitsGradient := NewTensor3D(
		input.Batch,
		input.Time,
		m.OutputSize,
	)

	if logitsGradient == nil {
		return nil, ErrInvalidSTTModelBackward
	}

	targetOffset := 0

	for batch := 0; batch < input.Batch; batch++ {

		targetLength := targetLengths[batch]

		target := targets[targetOffset : targetOffset+targetLength]

		singleLogits := NewTensor3D(
			1,
			input.Time,
			m.OutputSize,
		)

		if singleLogits == nil {
			return nil, ErrInvalidSTTModelBackward
		}

		for time := 0; time < input.Time; time++ {
			for feature := 0; feature < m.OutputSize; feature++ {

				value, err := logits.Get(
					batch,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				if err := singleLogits.Set(
					0,
					time,
					feature,
					value,
				); err != nil {
					return nil, err
				}
			}
		}

		gradient, err := CTCGradient(
			singleLogits,
			target,
			inputLengths[batch],
			m.CTC.BlankID,
		)

		if err != nil {
			return nil, err
		}

		if gradient == nil {
			return nil, ErrInvalidSTTModelBackward
		}

		for time := 0; time < input.Time; time++ {
			for feature := 0; feature < m.OutputSize; feature++ {

				value, err := gradient.Get(
					0,
					time,
					feature,
				)

				if err != nil {
					return nil, err
				}

				if err := logitsGradient.Set(
					batch,
					time,
					feature,
					value,
				); err != nil {
					return nil, err
				}
			}
		}

		targetOffset += targetLength
	}

	/*
		Linear backward:

		    logits
		      ↓
		    Linear
		      ↓
		    encoder gradient
	*/

	encoderGradient, decoderGradient, err :=
		m.Decoder.BackwardTensor(
			encoderOutput,
			logitsGradient,
		)

	if err != nil {
		return nil, err
	}

	if encoderGradient == nil ||
		decoderGradient == nil {
		return nil, ErrInvalidSTTModelBackward
	}

	/*
		BiLSTM backward:

		    encoder gradient
		          ↓
		    BiLSTM.Backward
		          ↓
		    input gradient
	*/

	encoderResult, err := m.Encoder.Backward(
		input,
		inputLengths,
		encoderGradient,
	)

	if err != nil {
		return nil, err
	}

	if encoderResult == nil {
		return nil, ErrInvalidSTTModelBackward
	}

	return &STTModelGradient{
		Encoder: encoderResult,
		Decoder: decoderGradient,
	}, nil
}
func (m *STTModel) Update(
	gradient *STTModelGradient,
	optimizer Optimizer,
) error {

	if m == nil ||
		gradient == nil ||
		optimizer == nil {
		return ErrInvalidSTTTraining
	}

	parameters := []*Parameter{
		{
			Name:      "decoder.weights",
			Values:    m.Decoder.Weights,
			Gradients: gradient.Decoder.Weights,
		},
		{
			Name:      "decoder.bias",
			Values:    m.Decoder.Bias,
			Gradients: gradient.Decoder.Bias,
		},
		{
			Name:   "encoder.forward_lstm.weights",
			Values: m.Encoder.ForwardLSTM.Cell.Weights,
			Gradients: gradient.Encoder.
				ForwardParameters.Weights,
		},
		{
			Name:   "encoder.forward_lstm.bias",
			Values: m.Encoder.ForwardLSTM.Cell.Bias,
			Gradients: gradient.Encoder.
				ForwardParameters.Bias,
		},
		{
			Name:   "encoder.backward_lstm.weights",
			Values: m.Encoder.BackwardLSTM.Cell.Weights,
			Gradients: gradient.Encoder.
				BackwardParameters.Weights,
		},
		{
			Name:   "encoder.backward_lstm.bias",
			Values: m.Encoder.BackwardLSTM.Cell.Bias,
			Gradients: gradient.Encoder.
				BackwardParameters.Bias,
		},
	}

	return optimizer.Step(parameters)
}
func CreateBatch(
	dataset *Dataset,
	start int,
	batchSize int,
) (*Batch, error) {
	if dataset == nil {
		return nil, ErrInvalidBatch
	}

	if batchSize <= 0 {
		return nil, ErrBatchSizeInvalid
	}

	if start < 0 || start >= dataset.Len() {
		return nil, ErrEmptyBatch
	}

	end := start + batchSize

	if end > dataset.Len() {
		end = dataset.Len()
	}

	count := end - start

	if count <= 0 {
		return nil, ErrEmptyBatch
	}

	// Find maximum sequence length.
	maxTime := 0
	featureDim := 0

	for i := start; i < end; i++ {
		sample, err := dataset.Get(i)
		if err != nil {
			return nil, err
		}

		if sample.Features == nil {
			return nil, ErrInvalidBatch
		}

		if sample.Features.Rows > maxTime {
			maxTime = sample.Features.Rows
		}

		if featureDim == 0 {
			featureDim = sample.Features.Cols
		}

		if sample.Features.Cols != featureDim {
			return nil, ErrInvalidBatch
		}
	}

	if maxTime <= 0 || featureDim <= 0 {
		return nil, ErrInvalidBatch
	}

	// Batch layout:
	//
	// [batch][time][feature]
	//
	// Flattened into:
	//
	// [(batch * maxTime) x featureDim]
	features := NewFeatureMatrix(
		count*maxTime,
		featureDim,
	)

	if features == nil {
		return nil, ErrInvalidBatch
	}

	labels := make([]int, 0)

	inputLengths := make([]int, count)
	labelLengths := make([]int, count)

	for batchIndex := 0; batchIndex < count; batchIndex++ {
		sample, err := dataset.Get(start + batchIndex)
		if err != nil {
			return nil, err
		}

		inputLengths[batchIndex] = sample.Features.Rows
		labelLengths[batchIndex] = len(sample.Labels)

		for row := 0; row < sample.Features.Rows; row++ {
			for col := 0; col < featureDim; col++ {
				features.Set(
					batchIndex*maxTime+row,
					col,
					sample.Features.Get(row, col),
				)
			}
		}

		labels = append(
			labels,
			sample.Labels...,
		)
	}

	return &Batch{
		Features:     features,
		Labels:       labels,
		InputLengths: inputLengths,
		LabelLengths: labelLengths,
		BatchSize:    count,
		TimeSteps:    maxTime,
		FeatureDim:   featureDim,
	}, nil
}
func (m *STTModel) TrainStep(
	input *Tensor3D,
	inputLengths []int,
	targets []int,
	targetLengths []int,
	optimizer Optimizer,
) (*STTTrainResult, error) {

	if m == nil ||
		m.Encoder == nil ||
		m.Decoder == nil ||
		m.CTC == nil {
		return nil, ErrInvalidSTTTraining
	}

	if input == nil ||
		optimizer == nil {
		return nil, ErrInvalidSTTTraining
	}

	if len(inputLengths) != input.Batch ||
		len(targetLengths) != input.Batch {
		return nil, ErrInvalidSTTTraining
	}

	// Forward

	encoderOutput, err := m.Encoder.Forward(
		input,
		inputLengths,
	)
	if err != nil {
		return nil, err
	}

	logits, err := m.Decoder.ForwardTensor(
		encoderOutput,
	)
	if err != nil {
		return nil, err
	}

	// Loss

	loss, err := CTCLossBatch(
		logits,
		targets,
		inputLengths,
		targetLengths,
		m.CTC.BlankID,
	)
	if err != nil {
		return nil, err
	}

	// Backward

	gradient, err := m.Backward(
		input,
		inputLengths,
		targets,
		targetLengths,
	)
	if err != nil {
		return nil, err
	}

	// Update

	if err := m.Update(
		gradient,
		optimizer,
	); err != nil {
		return nil, err
	}

	return &STTTrainResult{
		Loss: loss,
	}, nil
}
func NewSGD(learningRate float32) (*SGD, error) {

	if learningRate <= 0 {
		return nil, ErrInvalidLearningRate
	}

	return &SGD{
		LearningRate: learningRate,
	}, nil
}
func (o *SGD) Step(
	parameters []*Parameter,
) error {

	if o == nil {
		return ErrInvalidOptimizer
	}

	if len(parameters) == 0 {
		return ErrInvalidOptimizer
	}

	for _, parameter := range parameters {

		if parameter == nil ||
			len(parameter.Values) == 0 ||
			len(parameter.Values) != len(parameter.Gradients) {
			return ErrInvalidParameter
		}

		for i := range parameter.Values {
			parameter.Values[i] -=
				o.LearningRate * parameter.Gradients[i]
		}
	}

	return nil
}
func NewParameter(
	name string,
	values []float32,
	gradients []float32,
) (*Parameter, error) {

	if name == "" ||
		len(values) == 0 ||
		len(values) != len(gradients) {
		return nil, ErrInvalidParameter
	}

	return &Parameter{
		Name:      name,
		Values:    values,
		Gradients: gradients,
	}, nil
}
func NewAdam(learningRate float32) (*Adam, error) {
	if learningRate <= 0 {
		return nil, ErrInvalidAdam
	}

	return &Adam{
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		States:       make(map[string]*AdamState),
	}, nil
}
func (o *Adam) Step(
	parameters []*Parameter,
) error {

	if o == nil {
		return ErrInvalidAdam
	}

	if len(parameters) == 0 {
		return ErrInvalidAdam
	}

	for _, parameter := range parameters {
		if parameter == nil ||
			parameter.Name == "" ||
			len(parameter.Values) == 0 ||
			len(parameter.Values) != len(parameter.Gradients) {
			return ErrInvalidParameter
		}
	}

	o.StepCount++

	beta1Correction := float32(
		1.0 - math.Pow(
			float64(o.Beta1),
			float64(o.StepCount),
		),
	)

	beta2Correction := float32(
		1.0 - math.Pow(
			float64(o.Beta2),
			float64(o.StepCount),
		),
	)

	for _, parameter := range parameters {

		state, exists := o.States[parameter.Name]

		if !exists {
			state = &AdamState{
				FirstMoment:  make([]float32, len(parameter.Values)),
				SecondMoment: make([]float32, len(parameter.Values)),
			}

			o.States[parameter.Name] = state
		}

		if len(state.FirstMoment) != len(parameter.Values) ||
			len(state.SecondMoment) != len(parameter.Values) {
			return ErrInvalidAdam
		}

		for i := range parameter.Values {

			gradient := parameter.Gradients[i]

			state.FirstMoment[i] =
				o.Beta1*state.FirstMoment[i] +
					(1-o.Beta1)*gradient

			state.SecondMoment[i] =
				o.Beta2*state.SecondMoment[i] +
					(1-o.Beta2)*gradient*gradient

			firstMomentHat :=
				state.FirstMoment[i] / beta1Correction

			secondMomentHat :=
				state.SecondMoment[i] / beta2Correction

			parameter.Values[i] -=
				o.LearningRate *
					firstMomentHat /
					(float32(math.Sqrt(
						float64(secondMomentHat),
					)) + o.Epsilon)
		}
	}

	return nil
}
func (m *STTModel) Save(path string) error {

	if m == nil ||
		m.Encoder == nil ||
		m.Decoder == nil {
		return ErrInvalidModelFile
	}

	if path == "" {
		return ErrInvalidModelFile
	}

	file := STTModelFile{
		Version:             1,
		InputSize:           m.InputSize,
		HiddenSize:          m.HiddenSize,
		OutputSize:          m.OutputSize,
		VocabularyTokens:    m.CTC.Vocabulary.Tokens(),
		VocabularyBlankID:   m.CTC.BlankID,
		DecoderWeights:      m.Decoder.Weights,
		DecoderBias:         m.Decoder.Bias,
		ForwardLSTMWeights:  m.Encoder.ForwardLSTM.Cell.Weights,
		ForwardLSTMBias:     m.Encoder.ForwardLSTM.Cell.Bias,
		BackwardLSTMWeights: m.Encoder.BackwardLSTM.Cell.Weights,
		BackwardLSTMBias:    m.Encoder.BackwardLSTM.Cell.Bias,
	}

	data, err := json.Marshal(file)
	if err != nil {
		return err
	}

	return os.WriteFile(
		path,
		data,
		0644,
	)
}
func (v *Vocabulary) Tokens() []string {
	if v == nil {
		return nil
	}

	return append([]string(nil), v.tokens...)
}
func (v *Vocabulary) EncodeTokens() []string {
	return v.Tokens()
}

// DataLoader

func NewDataLoader(
	dataset *Dataset,
	batchSize int,
	shuffle bool,
	rng *rand.Rand,
) (*DataLoader, error) {
	if dataset == nil || dataset.Len() == 0 {
		return nil, ErrInvalidDataLoader
	}

	if batchSize <= 0 {
		return nil, ErrInvalidDataLoader
	}

	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	indices := make([]int, dataset.Len())

	for i := range indices {
		indices[i] = i
	}

	loader := &DataLoader{
		Dataset:   dataset,
		BatchSize: batchSize,
		Shuffle:   shuffle,
		RNG:       rng,
		indices:   indices,
	}

	loader.Reset()

	return loader, nil
}
func (l *DataLoader) Reset() {
	if l == nil {
		return
	}

	l.position = 0

	if len(l.indices) != l.Dataset.Len() {
		l.indices = make([]int, l.Dataset.Len())

		for i := range l.indices {
			l.indices[i] = i
		}
	}

	if l.Shuffle {
		l.RNG.Shuffle(len(l.indices), func(i, j int) {
			l.indices[i], l.indices[j] = l.indices[j], l.indices[i]
		})
	}
}
func (l *DataLoader) Next() (*Batch, bool, error) {
	if l == nil || l.Dataset == nil {
		return nil, false, ErrInvalidDataLoader
	}

	if l.position >= len(l.indices) {
		return nil, false, nil
	}

	end := l.position + l.BatchSize
	if end > len(l.indices) {
		end = len(l.indices)
	}

	count := end - l.position

	samples := make([]TrainingSample, count)

	for i := 0; i < count; i++ {
		index := l.indices[l.position+i]

		sample, err := l.Dataset.Get(index)
		if err != nil {
			return nil, false, err
		}

		samples[i] = *sample
	}

	batchDataset := &Dataset{
		Samples: samples,
	}

	batch, err := CreateBatch(
		batchDataset,
		0,
		count,
	)
	if err != nil {
		return nil, false, err
	}

	l.position = end

	return batch, true, nil
}

// DataLoader
func (m *STTModel) EvaluateBatch(
	input *Tensor3D,
	inputLengths []int,
	targets []int,
	targetLengths []int,
) (float32, error) {

	if m == nil ||
		m.Encoder == nil ||
		m.Decoder == nil ||
		m.CTC == nil {
		return 0, ErrInvalidSTTModel
	}

	if input == nil {
		return 0, ErrInvalidTensor
	}

	if len(inputLengths) != input.Batch {
		return 0, ErrInvalidSTTModel
	}

	if len(targetLengths) != input.Batch {
		return 0, ErrInvalidSTTModel
	}

	totalTargets := 0

	for _, length := range targetLengths {
		if length < 0 {
			return 0, ErrInvalidSTTModel
		}

		totalTargets += length
	}

	if totalTargets != len(targets) {
		return 0, ErrInvalidSTTModel
	}

	encoded, err := m.Encoder.Forward(
		input,
		inputLengths,
	)
	if err != nil {
		return 0, err
	}

	logits, err := m.Decoder.ForwardTensor(
		encoded,
	)
	if err != nil {
		return 0, err
	}

	loss, err := CTCLossBatch(
		logits,
		targets,
		inputLengths,
		targetLengths,
		m.CTC.BlankID,
	)
	if err != nil {
		return 0, err
	}

	return loss, nil
}
func (m *STTModel) EvaluateBatchResult(
	input *Tensor3D,
	inputLengths []int,
	targets []int,
	targetLengths []int,
) (*EvaluationResult, error) {

	if m == nil ||
		m.Encoder == nil ||
		m.Decoder == nil ||
		m.CTC == nil {
		return nil, ErrInvalidSTTModel
	}

	if input == nil {
		return nil, ErrInvalidSTTModel
	}

	if len(inputLengths) != input.Batch {
		return nil, ErrInvalidSTTModel
	}

	if len(targetLengths) != input.Batch {
		return nil, ErrInvalidSTTModel
	}

	totalTargets := 0

	for _, length := range targetLengths {
		if length < 0 {
			return nil, ErrInvalidSTTModel
		}

		totalTargets += length
	}

	if totalTargets != len(targets) {
		return nil, ErrInvalidSTTModel
	}

	encoded, err := m.Encoder.Forward(
		input,
		inputLengths,
	)
	if err != nil {
		return nil, err
	}

	logits, err := m.Decoder.ForwardTensor(encoded)
	if err != nil {
		return nil, err
	}

	loss, err := CTCLossBatch(
		logits,
		targets,
		inputLengths,
		targetLengths,
		m.CTC.BlankID,
	)
	if err != nil {
		return nil, err
	}

	predictions, err := m.CTC.Decode(
		logits,
		inputLengths,
	)
	if err != nil {
		return nil, err
	}

	return &EvaluationResult{
		Loss:        loss,
		Predictions: predictions,
	}, nil
}
func (v *Vocabulary) DecodeIDs(ids []int) (string, error) {
	if v == nil {
		return "", ErrEmptyVocabulary
	}

	var result strings.Builder

	for _, id := range ids {
		token, err := v.DecodeID(id)
		if err != nil {
			return "", err
		}

		if id == v.blankID {
			continue
		}

		result.WriteString(token)
	}

	return result.String(), nil
}

// Trainer
func NewTrainer(
	model *STTModel,
	optimizer Optimizer,
) (*Trainer, error) {

	if model == nil ||
		optimizer == nil {
		return nil, ErrInvalidTrainer
	}

	return &Trainer{
		Model:     model,
		Optimizer: optimizer,
		RNG: rand.New(
			rand.NewSource(42),
		),
	}, nil
}
func (t *Trainer) TrainEpoch(loader *DataLoader) (float32, error) {
	if t == nil || t.Model == nil || t.Optimizer == nil {
		return 0, ErrInvalidTrainer
	}

	if loader == nil {
		return 0, ErrInvalidDataLoader
	}

	loader.Reset()

	var totalLoss float32
	var batchCount int

	for {
		batch, ok, err := loader.Next()
		if err != nil {
			return 0, err
		}

		if !ok {
			break
		}

		loss, err := t.TrainBatch(batch, batch.Labels)
		if err != nil {
			return 0, err
		}

		totalLoss += loss
		batchCount++
	}

	if batchCount == 0 {
		return 0, ErrEmptyDataset
	}

	return totalLoss / float32(batchCount), nil
}
func (t *Trainer) TrainBatch(
	batch *Batch,
	targets []int,
) (float32, error) {
	if t == nil ||
		t.Model == nil ||
		t.Optimizer == nil ||
		batch == nil {
		return 0, ErrInvalidTrainer
	}
	input, err := batch.ToTensor()
	if err != nil {
		return 0, err
	}
	result, err := t.Model.TrainStep(
		input,
		batch.InputLengths,
		targets,
		batch.LabelLengths,
		t.Optimizer,
	)

	if err != nil {
		return 0, err
	}

	return result.Loss, nil
}
func (t *Trainer) Train(
	dataset *Dataset,
	config TrainingConfig,
) ([]float32, error) {

	if t == nil ||
		t.Model == nil ||
		t.Optimizer == nil {
		return nil, ErrInvalidSTTTrainingIntegration
	}

	if dataset == nil ||
		dataset.Len() == 0 {
		return nil, ErrEmptyTrainingDataset
	}

	if config.Epochs <= 0 ||
		config.BatchSize <= 0 {
		return nil, ErrInvalidSTTTrainingIntegration
	}
	if config.CheckpointEvery > 0 && strings.TrimSpace(config.CheckpointPath) == "" {
		return nil, ErrInvalidTrainingConfig
	}
	loader, err := NewDataLoader(
		dataset,
		config.BatchSize,
		true,
		t.RNG,
	)
	if err != nil {
		return nil, err
	}

	losses := make(
		[]float32,
		config.Epochs,
	)

	for epoch := 0; epoch < config.Epochs; epoch++ {

		loss, err := t.TrainEpoch(loader)
		if err != nil {
			return nil, err
		}

		losses[epoch] = loss
	}

	return losses, nil
}
func (t *Trainer) Evaluate(
	loader *DataLoader,
) (float32, error) {

	if t == nil ||
		t.Model == nil ||
		t.Optimizer == nil {
		return 0, ErrInvalidTrainer
	}

	if loader == nil {
		return 0, ErrInvalidDataLoader
	}

	loader.Reset()

	var totalLoss float32
	var batchCount int

	for {
		batch, ok, err := loader.Next()
		if err != nil {
			return 0, err
		}

		if !ok {
			break
		}

		input, err := batch.ToTensor()
		if err != nil {
			return 0, err
		}

		loss, err := t.Model.EvaluateBatch(
			input,
			batch.InputLengths,
			batch.Labels,
			batch.LabelLengths,
		)
		if err != nil {
			return 0, err
		}

		totalLoss += loss
		batchCount++
	}

	if batchCount == 0 {
		return 0, ErrEmptyDataset
	}

	return totalLoss / float32(batchCount), nil
}
func (t *Trainer) EvaluateWithMetrics(
	loader *DataLoader,
) (*TrainerEvaluationResult, error) {

	if t == nil ||
		t.Model == nil ||
		t.Optimizer == nil {
		return nil, ErrInvalidTrainer
	}

	if loader == nil {
		return nil, ErrInvalidDataLoader
	}

	loader.Reset()

	var totalLoss float32
	var totalCER float32
	var totalWER float32
	var sampleCount int
	var batchCount int

	for {
		batch, ok, err := loader.Next()
		if err != nil {
			return nil, err
		}

		if !ok {
			break
		}

		input, err := batch.ToTensor()
		if err != nil {
			return nil, err
		}

		result, err := t.Model.EvaluateBatchResult(
			input,
			batch.InputLengths,
			batch.Labels,
			batch.LabelLengths,
		)
		if err != nil {
			return nil, err
		}

		if len(result.Predictions) != batch.BatchSize {
			return nil, ErrInvalidTrainer
		}

		targetOffset := 0

		for i := 0; i < batch.BatchSize; i++ {

			labelLength := batch.LabelLengths[i]

			if targetOffset+labelLength > len(batch.Labels) {
				return nil, ErrInvalidTrainer
			}

			targetIDs := batch.Labels[targetOffset : targetOffset+labelLength]

			reference, err := t.Model.CTC.Vocabulary.DecodeIDs(
				targetIDs,
			)
			if err != nil {
				return nil, err
			}

			hypothesis := result.Predictions[i]

			cer, err := CharacterErrorRate(
				reference,
				hypothesis,
			)
			if err != nil {
				return nil, err
			}

			wer, err := WordErrorRate(
				reference,
				hypothesis,
			)
			if err != nil {
				return nil, err
			}

			totalCER += cer
			totalWER += wer

			sampleCount++
			targetOffset += labelLength
		}

		if targetOffset != len(batch.Labels) {
			return nil, ErrInvalidTrainer
		}

		totalLoss += result.Loss
		batchCount++
	}

	if batchCount == 0 || sampleCount == 0 {
		return nil, ErrEmptyDataset
	}

	return &TrainerEvaluationResult{
		Loss: totalLoss / float32(batchCount),
		CER:  totalCER / float32(sampleCount),
		WER:  totalWER / float32(sampleCount),
	}, nil
}
func (t *Trainer) TrainWithValidation(
	trainDataset *Dataset,
	validationDataset *Dataset,
	config TrainingConfig,
) (*TrainingHistory, error) {
	return t.trainWithValidationFromEpoch(
		trainDataset,
		validationDataset,
		config,
		0,
		nil,
		0,
		-1,
		nil,
	)
}
func (t *Trainer) ResumeWithValidation(
	checkpointPath string,
	trainDataset *Dataset,
	validationDataset *Dataset,
	config TrainingConfig,
) (*TrainingHistory, error) {

	if t == nil ||
		t.Model == nil ||
		t.Optimizer == nil {
		return nil, ErrInvalidTrainer
	}

	checkpoint, err := LoadTrainingCheckpoint(checkpointPath)
	if err != nil {
		return nil, err
	}

	if err := t.RestoreCheckpoint(checkpoint); err != nil {
		return nil, err
	}

	if config.Epochs <= checkpoint.Epoch {
		return nil, ErrInvalidSTTTrainingIntegration
	}

	return t.trainWithValidationFromEpoch(
		trainDataset,
		validationDataset,
		config,
		checkpoint.Epoch,
		checkpoint.BestModel,
		checkpoint.BestLoss,
		checkpoint.BestEpoch,
		checkpoint.EarlyStopping,
	)
}
func (t *Trainer) trainWithValidationFromEpoch(
	trainDataset *Dataset,
	validationDataset *Dataset,
	config TrainingConfig,
	startEpoch int,
	bestSnapshot *STTModelSnapshot,
	bestLoss float32,
	bestEpoch int,
	earlyStoppingState *EarlyStoppingState,
) (*TrainingHistory, error) {

	if t == nil ||
		t.Model == nil ||
		t.Optimizer == nil {
		return nil, ErrInvalidTrainer
	}

	if trainDataset == nil ||
		trainDataset.Len() == 0 ||
		validationDataset == nil ||
		validationDataset.Len() == 0 {
		return nil, ErrEmptyTrainingDataset
	}

	if config.Epochs <= 0 ||
		config.BatchSize <= 0 {
		return nil, ErrInvalidSTTTrainingIntegration
	}

	if config.CheckpointEvery > 0 &&
		strings.TrimSpace(config.CheckpointPath) == "" {
		return nil, ErrInvalidTrainingConfig
	}
	trainLoader, err := NewDataLoader(
		trainDataset,
		config.BatchSize,
		true,
		t.RNG,
	)
	if err != nil {
		return nil, err
	}

	validationLoader, err := NewDataLoader(
		validationDataset,
		config.BatchSize,
		false,
		nil,
	)
	if err != nil {
		return nil, err
	}

	var earlyStopping *EarlyStopping

	if config.EarlyStoppingPatience > 0 {
		var err error

		earlyStopping, err = NewEarlyStopping(
			config.EarlyStoppingPatience,
			config.EarlyStoppingMinDelta,
		)
		if err != nil {
			return nil, err
		}

		if earlyStoppingState != nil {
			if err := earlyStopping.Restore(earlyStoppingState); err != nil {
				return nil, err
			}
		}
	}

	if bestSnapshot == nil {
		bestLoss = float32(0)
		bestEpoch = -1
	}
	history := &TrainingHistory{
		TrainLoss:      make([]float32, 0, config.Epochs),
		ValidationLoss: make([]float32, 0, config.Epochs),
		ValidationCER:  make([]float32, 0, config.Epochs),
		ValidationWER:  make([]float32, 0, config.Epochs),
	}

	for epoch := startEpoch; epoch < config.Epochs; epoch++ {
		trainLoss, err := t.TrainEpoch(trainLoader)
		if err != nil {
			return nil, err
		}

		evaluation, err := t.EvaluateWithMetrics(
			validationLoader,
		)
		if err != nil {
			return nil, err
		}

		if bestEpoch == -1 ||
			evaluation.Loss < bestLoss {

			snapshot, err := t.Model.Snapshot()
			if err != nil {
				return nil, err
			}

			bestSnapshot = snapshot
			bestLoss = evaluation.Loss
			bestEpoch = epoch
		}

		history.TrainLoss = append(
			history.TrainLoss,
			trainLoss,
		)

		history.ValidationLoss = append(
			history.ValidationLoss,
			evaluation.Loss,
		)

		history.ValidationCER = append(
			history.ValidationCER,
			evaluation.CER,
		)

		history.ValidationWER = append(
			history.ValidationWER,
			evaluation.WER,
		)
		if earlyStopping != nil {
			stop, err := earlyStopping.Update(
				evaluation.Loss,
			)
			if err != nil {
				return nil, err
			}

			if stop {
				break
			}
		}
		if config.CheckpointEvery > 0 &&
			(epoch+1)%config.CheckpointEvery == 0 {

			step := 0

			switch optimizer := t.Optimizer.(type) {
			case *Adam:
				step = optimizer.StepCount
			default:
				return nil, ErrInvalidTrainingCheckpoint
			}

			checkpoint, err := t.CreateTrainingCheckpoint(
				epoch+1,
				step,
				bestSnapshot,
				bestLoss,
				bestEpoch,
				earlyStopping,
			)
			if err != nil {
				return nil, err
			}

			if err := SaveTrainingCheckpoint(
				config.CheckpointPath,
				checkpoint,
			); err != nil {
				return nil, err
			}
		}
	}

	if bestSnapshot == nil {
		return nil, ErrEmptyDataset
	}

	if err := t.Model.Restore(bestSnapshot); err != nil {
		return nil, err
	}

	history.BestEpoch = bestEpoch
	history.BestLoss = bestLoss

	return history, nil
}
func (t *Trainer) CreateCheckpoint(
	epoch int,
	step int,
) (*TrainingCheckpoint, error) {

	if t == nil ||
		t.Model == nil ||
		t.Optimizer == nil {
		return nil, ErrInvalidTrainer
	}

	if epoch < 0 || step < 0 {
		return nil, ErrInvalidTrainingCheckpoint
	}

	modelSnapshot, err := t.Model.Snapshot()
	if err != nil {
		return nil, err
	}

	var optimizerSnapshot *OptimizerCheckpoint

	switch optimizer := t.Optimizer.(type) {
	case *Adam:
		optimizerSnapshot, err = optimizer.Snapshot()
		if err != nil {
			return nil, err
		}

	default:
		return nil, ErrInvalidTrainingCheckpoint
	}

	return &TrainingCheckpoint{
		Version: 1,

		Epoch: epoch,
		Step:  step,

		Model:     modelSnapshot,
		Optimizer: optimizerSnapshot,
	}, nil
}
func (t *Trainer) RestoreCheckpoint(
	checkpoint *TrainingCheckpoint,
) error {

	if t == nil ||
		t.Model == nil ||
		t.Optimizer == nil {
		return ErrInvalidTrainer
	}
	if checkpoint == nil ||
		(checkpoint.Version != 1 &&
			checkpoint.Version != 2) ||
		checkpoint.Model == nil ||
		checkpoint.Optimizer == nil {
		return ErrInvalidTrainingCheckpoint
	}

	if err := t.Model.Restore(
		checkpoint.Model,
	); err != nil {
		return err
	}

	switch optimizer := t.Optimizer.(type) {
	case *Adam:
		if err := optimizer.Restore(
			checkpoint.Optimizer,
		); err != nil {
			return err
		}

	default:
		return ErrInvalidTrainingCheckpoint
	}

	return nil
}
func (t *Trainer) SaveCheckpoint(path string, epoch, step int) error {
	if t == nil {
		return ErrInvalidTrainer
	}

	checkpoint, err := t.CreateCheckpoint(epoch, step)
	if err != nil {
		return err
	}

	return SaveTrainingCheckpoint(path, checkpoint)
}
func (t *Trainer) LoadCheckpoint(path string) error {
	if t == nil {
		return ErrInvalidTrainer
	}

	checkpoint, err := LoadTrainingCheckpoint(path)
	if err != nil {
		return err
	}

	return t.RestoreCheckpoint(checkpoint)
}
func (t *Trainer) CreateTrainingCheckpoint(
	epoch int,
	step int,
	bestModel *STTModelSnapshot,
	bestLoss float32,
	bestEpoch int,
	earlyStopping *EarlyStopping,
) (*TrainingCheckpoint, error) {

	if t == nil ||
		t.Model == nil ||
		t.Optimizer == nil {
		return nil, ErrInvalidTrainer
	}

	if epoch < 0 || step < 0 {
		return nil, ErrInvalidTrainingCheckpoint
	}

	modelSnapshot, err := t.Model.Snapshot()
	if err != nil {
		return nil, err
	}

	var optimizerCheckpoint *OptimizerCheckpoint

	switch optimizer := t.Optimizer.(type) {
	case *Adam:
		optimizerCheckpoint, err = optimizer.Snapshot()
		if err != nil {
			return nil, err
		}
	default:
		return nil, ErrInvalidTrainingCheckpoint
	}

	checkpoint := &TrainingCheckpoint{
		Version:   2,
		Epoch:     epoch,
		Step:      step,
		Model:     modelSnapshot,
		Optimizer: optimizerCheckpoint,
		BestModel: nil,
		BestLoss:  bestLoss,
		BestEpoch: bestEpoch,
	}

	if bestModel != nil {
		checkpoint.BestModel = &STTModelSnapshot{
			DecoderWeights:      append([]float32(nil), bestModel.DecoderWeights...),
			DecoderBias:         append([]float32(nil), bestModel.DecoderBias...),
			ForwardLSTMWeights:  append([]float32(nil), bestModel.ForwardLSTMWeights...),
			ForwardLSTMBias:     append([]float32(nil), bestModel.ForwardLSTMBias...),
			BackwardLSTMWeights: append([]float32(nil), bestModel.BackwardLSTMWeights...),
			BackwardLSTMBias:    append([]float32(nil), bestModel.BackwardLSTMBias...),
		}
	}

	if earlyStopping != nil {
		state, err := earlyStopping.Snapshot()
		if err != nil {
			return nil, err
		}

		checkpoint.EarlyStopping = state
	}

	return checkpoint, nil
}

// Trainer
func NewEarlyStopping(
	patience int,
	minDelta float32,
) (*EarlyStopping, error) {

	if patience <= 0 {
		return nil, ErrInvalidEarlyStopping
	}

	if minDelta < 0 {
		return nil, ErrInvalidEarlyStopping
	}

	return &EarlyStopping{
		Patience: patience,
		MinDelta: minDelta,
	}, nil
}
func (e *EarlyStopping) Update(
	loss float32,
) (bool, error) {

	if e == nil {
		return false, ErrInvalidEarlyStopping
	}

	if loss < 0 {
		return false, ErrInvalidEarlyStopping
	}

	if !e.initialized {
		e.bestLoss = loss
		e.waitCount = 0
		e.initialized = true

		return false, nil
	}

	if loss < e.bestLoss-e.MinDelta {
		e.bestLoss = loss
		e.waitCount = 0

		return false, nil
	}

	e.waitCount++

	if e.waitCount >= e.Patience {
		return true, nil
	}

	return false, nil
}
func (e *EarlyStopping) BestLoss() float32 {
	if e == nil || !e.initialized {
		return 0
	}

	return e.bestLoss
}
func (e *EarlyStopping) WaitCount() int {
	if e == nil {
		return 0
	}

	return e.waitCount
}
func (m *STTModel) Snapshot() (*STTModelSnapshot, error) {
	if m == nil ||
		m.Encoder == nil ||
		m.Decoder == nil {
		return nil, ErrInvalidSTTModel
	}

	return &STTModelSnapshot{
		DecoderWeights: append(
			[]float32(nil),
			m.Decoder.Weights...,
		),
		DecoderBias: append(
			[]float32(nil),
			m.Decoder.Bias...,
		),

		ForwardLSTMWeights: append(
			[]float32(nil),
			m.Encoder.ForwardLSTM.Cell.Weights...,
		),
		ForwardLSTMBias: append(
			[]float32(nil),
			m.Encoder.ForwardLSTM.Cell.Bias...,
		),

		BackwardLSTMWeights: append(
			[]float32(nil),
			m.Encoder.BackwardLSTM.Cell.Weights...,
		),
		BackwardLSTMBias: append(
			[]float32(nil),
			m.Encoder.BackwardLSTM.Cell.Bias...,
		),
	}, nil
}
func (m *STTModel) Restore(
	snapshot *STTModelSnapshot,
) error {

	if m == nil ||
		m.Encoder == nil ||
		m.Decoder == nil ||
		snapshot == nil {
		return ErrInvalidSTTModel
	}

	if len(snapshot.DecoderWeights) != len(m.Decoder.Weights) ||
		len(snapshot.DecoderBias) != len(m.Decoder.Bias) ||
		len(snapshot.ForwardLSTMWeights) != len(m.Encoder.ForwardLSTM.Cell.Weights) ||
		len(snapshot.ForwardLSTMBias) != len(m.Encoder.ForwardLSTM.Cell.Bias) ||
		len(snapshot.BackwardLSTMWeights) != len(m.Encoder.BackwardLSTM.Cell.Weights) ||
		len(snapshot.BackwardLSTMBias) != len(m.Encoder.BackwardLSTM.Cell.Bias) {
		return ErrInvalidSTTModel
	}

	copy(
		m.Decoder.Weights,
		snapshot.DecoderWeights,
	)

	copy(
		m.Decoder.Bias,
		snapshot.DecoderBias,
	)

	copy(
		m.Encoder.ForwardLSTM.Cell.Weights,
		snapshot.ForwardLSTMWeights,
	)

	copy(
		m.Encoder.ForwardLSTM.Cell.Bias,
		snapshot.ForwardLSTMBias,
	)

	copy(
		m.Encoder.BackwardLSTM.Cell.Weights,
		snapshot.BackwardLSTMWeights,
	)

	copy(
		m.Encoder.BackwardLSTM.Cell.Bias,
		snapshot.BackwardLSTMBias,
	)

	return nil
}
func (o *Adam) Snapshot() (*OptimizerCheckpoint, error) {
	if o == nil {
		return nil, ErrInvalidAdam
	}

	states := make(
		map[string]*AdamState,
		len(o.States),
	)

	for name, state := range o.States {
		if state == nil {
			return nil, ErrInvalidAdam
		}

		states[name] = &AdamState{
			FirstMoment: append(
				[]float32(nil),
				state.FirstMoment...,
			),
			SecondMoment: append(
				[]float32(nil),
				state.SecondMoment...,
			),
		}
	}

	return &OptimizerCheckpoint{
		Type:      "adam",
		StepCount: o.StepCount,
		States:    states,
	}, nil
}
func (o *Adam) Restore(
	checkpoint *OptimizerCheckpoint,
) error {

	if o == nil ||
		checkpoint == nil {
		return ErrInvalidAdam
	}

	if checkpoint.Type != "adam" {
		return ErrInvalidAdam
	}

	if checkpoint.StepCount < 0 {
		return ErrInvalidAdam
	}

	states := make(
		map[string]*AdamState,
		len(checkpoint.States),
	)

	for name, state := range checkpoint.States {
		if state == nil {
			return ErrInvalidAdam
		}

		states[name] = &AdamState{
			FirstMoment: append(
				[]float32(nil),
				state.FirstMoment...,
			),
			SecondMoment: append(
				[]float32(nil),
				state.SecondMoment...,
			),
		}
	}

	o.StepCount = checkpoint.StepCount
	o.States = states

	return nil
}
func (e *EarlyStopping) Snapshot() (*EarlyStoppingState, error) {
	if e == nil {
		return nil, ErrInvalidEarlyStopping
	}

	return &EarlyStoppingState{
		BestLoss:    e.bestLoss,
		WaitCount:   e.waitCount,
		Initialized: e.initialized,
	}, nil
}
func (e *EarlyStopping) Restore(state *EarlyStoppingState) error {
	if e == nil || state == nil {
		return ErrInvalidEarlyStopping
	}

	if state.WaitCount < 0 {
		return ErrInvalidEarlyStopping
	}

	e.bestLoss = state.BestLoss
	e.waitCount = state.WaitCount
	e.initialized = state.Initialized

	return nil
}
func NewLocalSTT(
	model *STTModel,
	config FeatureConfig,
) (*LocalSTT, error) {

	if model == nil {
		return nil, ErrInvalidLocalSTT
	}

	if err := config.Validate(); err != nil {
		return nil, err
	}

	return &LocalSTT{
		Model:  model,
		Config: config,
	}, nil
}
func (s *LocalSTT) Transcribe(
	ctx context.Context,
	segment neurocall.AudioSegment,
) (neurocall.Transcript, error) {

	if s == nil || s.Model == nil {
		return neurocall.Transcript{}, ErrInvalidLocalSTT
	}

	if len(segment.Data) == 0 {
		return neurocall.Transcript{}, ErrEmptyAudio
	}

	select {
	case <-ctx.Done():
		return neurocall.Transcript{}, ctx.Err()
	default:
	}

	samples := make([]float32, len(segment.Data))

	for i, sample := range segment.Data {
		samples[i] = float32(sample) / 32768.0
	}

	config := s.Config

	if segment.SampleRate > 0 {
		config.SampleRate = segment.SampleRate
	}

	features, err := ExtractFeatures(
		samples,
		config,
	)
	if err != nil {
		return neurocall.Transcript{}, err
	}

	input := featureMatrixToTensor(features)

	if input == nil {
		return neurocall.Transcript{}, ErrInvalidLocalSTT
	}

	inputLengths := []int{
		features.Rows,
	}

	result, err := s.Model.Transcribe(
		input,
		inputLengths,
	)
	if err != nil {
		return neurocall.Transcript{}, err
	}

	if len(result) == 0 {
		return neurocall.Transcript{}, nil
	}

	return neurocall.Transcript{
		Text: result[0],
	}, nil
}
func NewCommonVoiceImporter(
	ffmpeg *media.FFmpeg,
	config FeatureConfig,
	tokenizer *Tokenizer,
) (*CommonVoiceImporter, error) {
	if ffmpeg == nil || tokenizer == nil {
		return nil, ErrInvalidCommonVoiceImporter
	}

	if err := config.Validate(); err != nil {
		return nil, err
	}

	return &CommonVoiceImporter{
		FFmpeg:        ffmpeg,
		FeatureConfig: config,
		Tokenizer:     tokenizer,
	}, nil
}
func LoadCommonVoiceManifest(
	path string,
) ([]CommonVoiceEntry, error) {
	if path == "" {
		return nil, ErrCommonVoiceManifest
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	// Some Common Voice lines can be long.
	scanner.Buffer(
		make([]byte, 64*1024),
		1024*1024,
	)

	headerLine := true
	pathIndex := -1
	sentenceIndex := -1

	entries := make([]CommonVoiceEntry, 0)

	for scanner.Scan() {
		line := scanner.Text()

		if headerLine {
			headerLine = false

			header := strings.Split(line, "\t")

			for i, column := range header {
				switch strings.TrimSpace(column) {
				case "path":
					pathIndex = i
				case "sentence":
					sentenceIndex = i
				}
			}

			if pathIndex < 0 || sentenceIndex < 0 {
				return nil, ErrCommonVoiceColumn
			}

			continue
		}

		if strings.TrimSpace(line) == "" {
			continue
		}

		fields := strings.Split(line, "\t")

		if pathIndex >= len(fields) ||
			sentenceIndex >= len(fields) {
			continue
		}

		audioPath := strings.TrimSpace(fields[pathIndex])
		text := strings.TrimSpace(fields[sentenceIndex])

		if audioPath == "" || text == "" {
			continue
		}

		entries = append(entries, CommonVoiceEntry{
			AudioPath: audioPath,
			Text:      text,
		})
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	if len(entries) == 0 {
		return nil, ErrCommonVoiceManifest
	}

	return entries, nil
}
func (i *CommonVoiceImporter) Import(
	ctx context.Context,
	datasetRoot string,
	manifestPath string,
	limit int,
) (*Dataset, error) {
	if i == nil ||
		i.FFmpeg == nil ||
		i.Tokenizer == nil {
		return nil, ErrInvalidCommonVoiceImporter
	}

	entries, err := LoadCommonVoiceManifest(manifestPath)
	if err != nil {
		return nil, err
	}

	return i.ImportEntries(
		ctx,
		datasetRoot,
		entries,
		limit,
	)
}
func SplitCommonVoice(
	entries []CommonVoiceEntry,
	validationRatio float64,
	rng *rand.Rand,
) (*CommonVoiceSplit, error) {
	if len(entries) == 0 {
		return nil, ErrInvalidCommonVoiceSplit
	}

	if validationRatio <= 0 ||
		validationRatio >= 1 {
		return nil, ErrInvalidCommonVoiceSplit
	}

	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	shuffled := append([]CommonVoiceEntry(nil), entries...)

	rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] =
			shuffled[j], shuffled[i]
	})

	validationCount := int(
		float64(len(shuffled)) * validationRatio,
	)

	if validationCount <= 0 {
		validationCount = 1
	}

	if validationCount >= len(shuffled) {
		validationCount = len(shuffled) - 1
	}

	trainCount := len(shuffled) - validationCount

	return &CommonVoiceSplit{
		Train: append(
			[]CommonVoiceEntry(nil),
			shuffled[:trainCount]...,
		),
		Valid: append(
			[]CommonVoiceEntry(nil),
			shuffled[trainCount:]...,
		),
	}, nil
}
func (i *CommonVoiceImporter) ImportEntries(
	ctx context.Context,
	datasetRoot string,
	entries []CommonVoiceEntry,
	limit int,
) (*Dataset, error) {
	if i == nil ||
		i.FFmpeg == nil ||
		i.Tokenizer == nil {
		return nil, ErrInvalidCommonVoiceImporter
	}

	if len(entries) == 0 {
		return nil, ErrEmptyDataset
	}

	if limit > 0 && limit < len(entries) {
		entries = entries[:limit]
	}

	dataset := NewDataset()

	for _, entry := range entries {
		audioPath := filepath.Join(
			datasetRoot,
			"clips",
			entry.AudioPath,
		)

		pcm, err := i.FFmpeg.ConvertToPCM16(
			ctx,
			audioPath,
			i.FeatureConfig.SampleRate,
			1,
		)
		if err != nil {
			return nil, err
		}

		samples, err := PCM16ToFloat32(pcm)
		if err != nil {
			return nil, err
		}

		features, err := ExtractFeatures(
			samples,
			i.FeatureConfig,
		)
		if err != nil {
			return nil, err
		}

		labels, err := i.Tokenizer.Encode(entry.Text)
		if err != nil {
			return nil, err
		}

		if err := dataset.Add(TrainingSample{
			Features: features,
			Labels:   labels,
			Text:     entry.Text,
		}); err != nil {
			return nil, err
		}
	}

	if dataset.Len() == 0 {
		return nil, ErrEmptyDataset
	}

	return dataset, nil
}

// Keep bufio imported while allowing future streaming extensions.
var _ = bufio.ErrInvalidUnreadByte
