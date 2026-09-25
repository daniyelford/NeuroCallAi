package stt

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/daniyelford/NeuroCallAi/pkg/media"
)

func TestPCM16ToFloat32(t *testing.T) {
	data := encodePCM16(
		0,
		32767,
		-32768,
	)

	samples, err := PCM16ToFloat32(data)
	if err != nil {
		t.Fatal(err)
	}

	t.Log(samples)
}
func TestPrepareFrames(t *testing.T) {
	sampleRate := 16000

	samples := make([]float32, sampleRate)

	for i := range samples {
		samples[i] = float32(
			math.Sin(
				2 * math.Pi *
					440 *
					float64(i) /
					float64(sampleRate),
			),
		)
	}

	frames := PrepareFrames(
		samples,
		sampleRate,
	)

	if len(frames) == 0 {
		t.Fatal("expected frames")
	}

	if len(frames[0]) != 400 {
		t.Fatalf(
			"expected frame size 400, got %d",
			len(frames[0]),
		)
	}

	t.Logf(
		"frames=%d frameSize=%d",
		len(frames),
		len(frames[0]),
	)
}
func TestExtractFeatures(t *testing.T) {

	config := DefaultFeatureConfig()

	sampleRate := config.SampleRate

	// duration := 1 * time.Second

	samples := make(
		[]float32,
		sampleRate,
	)

	for i := range samples {
		samples[i] = float32(
			math.Sin(
				2 * math.Pi *
					440 *
					float64(i) /
					float64(sampleRate),
			),
		)
	}

	matrix, err := ExtractFeatures(
		samples,
		config,
	)

	if err != nil {
		t.Fatal(err)
	}

	if matrix == nil {
		t.Fatal("matrix is nil")
	}

	t.Logf(
		"rows=%d cols=%d",
		matrix.Rows,
		matrix.Cols,
	)

	t.Logf(
		"first feature=%v",
		matrix.Get(0, 0),
	)

	if matrix.Cols != 80 {
		t.Fatalf(
			"expected 80 mel bins, got %d",
			matrix.Cols,
		)
	}

	if matrix.Rows == 0 {
		t.Fatal("expected frames")
	}
}
func TestTokenizer(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}

	ids, err := tokenizer.Encode("hello")
	if err != nil {
		t.Fatal(err)
	}

	t.Log("IDs:", ids)

	text, err := tokenizer.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("Decoded:", text)

	if text != "hello" {
		t.Fatalf(
			"expected hello, got %q",
			text,
		)
	}
}
func TestDataset(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}

	labels, err := tokenizer.Encode("hello")
	if err != nil {
		t.Fatal(err)
	}

	features := NewFeatureMatrix(98, 80)
	if features == nil {
		t.Fatal("failed to create feature matrix")
	}

	dataset := NewDataset()

	err = dataset.Add(TrainingSample{
		Features: features,
		Labels:   labels,
		Text:     "hello",
	})
	if err != nil {
		t.Fatal(err)
	}

	if dataset.Len() != 1 {
		t.Fatalf("expected dataset length 1, got %d", dataset.Len())
	}

	sample, err := dataset.Get(0)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("Text:", sample.Text)
	t.Log("Labels:", sample.Labels)
	t.Log("Features:", sample.Features.Rows, "x", sample.Features.Cols)
}
func createTestWAV(
	t *testing.T,
	path string,
	samples []int16,
) {
	t.Helper()

	const (
		sampleRate    = 16000
		channels      = 1
		bitsPerSample = 16
	)

	dataSize := len(samples) * 2
	fileSize := 36 + dataSize

	data := make([]byte, 44+dataSize)

	copy(data[0:4], []byte("RIFF"))
	binary.LittleEndian.PutUint32(data[4:8], uint32(fileSize))
	copy(data[8:12], []byte("WAVE"))

	copy(data[12:16], []byte("fmt "))
	binary.LittleEndian.PutUint32(data[16:20], 16)
	binary.LittleEndian.PutUint16(data[20:22], 1)
	binary.LittleEndian.PutUint16(data[22:24], channels)
	binary.LittleEndian.PutUint32(data[24:28], sampleRate)
	binary.LittleEndian.PutUint32(
		data[28:32],
		sampleRate*channels*bitsPerSample/8,
	)
	binary.LittleEndian.PutUint16(data[32:34], channels*bitsPerSample/8)
	binary.LittleEndian.PutUint16(data[34:36], bitsPerSample)

	copy(data[36:40], []byte("data"))
	binary.LittleEndian.PutUint32(data[40:44], uint32(dataSize))

	for i, sample := range samples {
		binary.LittleEndian.PutUint16(
			data[44+i*2:46+i*2],
			uint16(sample),
		)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
}
func TestLoadWAV(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.wav")

	createTestWAV(
		t,
		path,
		[]int16{
			0,
			32767,
			-32768,
			1000,
		},
	)

	audio, err := LoadWAV(path)
	if err != nil {
		t.Fatal(err)
	}

	if audio.SampleRate != 16000 {
		t.Fatalf(
			"expected sample rate 16000, got %d",
			audio.SampleRate,
		)
	}

	if audio.Channels != 1 {
		t.Fatalf(
			"expected channels 1, got %d",
			audio.Channels,
		)
	}

	if len(audio.Samples) != 4 {
		t.Fatalf(
			"expected 4 samples, got %d",
			len(audio.Samples),
		)
	}

	t.Log("Samples:", audio.Samples)
}
func TestLoadManifest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manifest.tsv")

	content := "" +
		"audio/0001.wav\thello world\n" +
		"audio/0002.wav\thow are you\n" +
		"audio/0003.wav\tgood morning\n"

	if err := os.WriteFile(
		path,
		[]byte(content),
		0644,
	); err != nil {
		t.Fatal(err)
	}

	entries, err := LoadManifest(path)
	if err != nil {
		t.Fatal(err)
	}

	if len(entries) != 3 {
		t.Fatalf(
			"expected 3 entries, got %d",
			len(entries),
		)
	}

	t.Log("Entries:", entries)

	if entries[0].AudioPath != "audio/0001.wav" {
		t.Fatalf(
			"unexpected audio path: %q",
			entries[0].AudioPath,
		)
	}

	if entries[0].Text != "hello world" {
		t.Fatalf(
			"unexpected text: %q",
			entries[0].Text,
		)
	}
}
func TestDatasetLoader(t *testing.T) {
	tempDir := t.TempDir()

	audioDir := filepath.Join(tempDir, "audio")

	if err := os.MkdirAll(audioDir, 0755); err != nil {
		t.Fatal(err)
	}

	audioPath := filepath.Join(audioDir, "0001.wav")

	samples := make([]int16, 16000)

	for i := range samples {
		samples[i] = int16(
			10000 * math.Sin(
				2*math.Pi*440*float64(i)/16000,
			),
		)
	}

	createTestWAV(
		t,
		audioPath,
		samples,
	)

	manifestPath := filepath.Join(
		tempDir,
		"manifest.tsv",
	)

	manifest := "audio/0001.wav\thello\n"

	if err := os.WriteFile(
		manifestPath,
		[]byte(manifest),
		0644,
	); err != nil {
		t.Fatal(err)
	}

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}

	loader, err := NewDatasetLoader(
		tokenizer,
		DefaultFeatureConfig(),
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset, err := loader.Load(manifestPath)
	if err != nil {
		t.Fatal(err)
	}

	if dataset.Len() != 1 {
		t.Fatalf(
			"expected 1 sample, got %d",
			dataset.Len(),
		)
	}

	sample, err := dataset.Get(0)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("Text:", sample.Text)
	t.Log("Labels:", sample.Labels)
	t.Log(
		"Features:",
		sample.Features.Rows,
		"x",
		sample.Features.Cols,
	)

	if sample.Text != "hello" {
		t.Fatalf(
			"expected hello, got %q",
			sample.Text,
		)
	}

	if len(sample.Labels) != 5 {
		t.Fatalf(
			"expected 5 labels, got %d",
			len(sample.Labels),
		)
	}

	if sample.Features.Cols != 80 {
		t.Fatalf(
			"expected 80 features, got %d",
			sample.Features.Cols,
		)
	}
}
func TestCreateBatch(t *testing.T) {
	dataset := NewDataset()

	f1 := NewFeatureMatrix(98, 80)
	f2 := NewFeatureMatrix(73, 80)

	err := dataset.Add(TrainingSample{
		Features: f1,
		Labels:   []int{8, 5, 12, 12, 15},
		Text:     "hello",
	})
	if err != nil {
		t.Fatal(err)
	}

	err = dataset.Add(TrainingSample{
		Features: f2,
		Labels:   []int{8, 15, 23},
		Text:     "how",
	})
	if err != nil {
		t.Fatal(err)
	}

	batch, err := CreateBatch(dataset, 0, 2)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("Batch size:", batch.BatchSize)
	t.Log("Time steps:", batch.TimeSteps)
	t.Log("Feature dim:", batch.FeatureDim)
	t.Log("Input lengths:", batch.InputLengths)
	t.Log("Label lengths:", batch.LabelLengths)
	t.Log("Feature rows:", batch.Features.Rows)
	t.Log("Feature cols:", batch.Features.Cols)

	if batch.BatchSize != 2 {
		t.Fatalf(
			"expected batch size 2, got %d",
			batch.BatchSize,
		)
	}

	if batch.TimeSteps != 98 {
		t.Fatalf(
			"expected max time 98, got %d",
			batch.TimeSteps,
		)
	}

	if batch.FeatureDim != 80 {
		t.Fatalf(
			"expected feature dim 80, got %d",
			batch.FeatureDim,
		)
	}

	if batch.Features.Rows != 196 {
		t.Fatalf(
			"expected 196 feature rows, got %d",
			batch.Features.Rows,
		)
	}

	if batch.Features.Cols != 80 {
		t.Fatalf(
			"expected 80 feature cols, got %d",
			batch.Features.Cols,
		)
	}

	if batch.InputLengths[0] != 98 ||
		batch.InputLengths[1] != 73 {
		t.Fatalf(
			"unexpected input lengths: %v",
			batch.InputLengths,
		)
	}

	if batch.LabelLengths[0] != 5 ||
		batch.LabelLengths[1] != 3 {
		t.Fatalf(
			"unexpected label lengths: %v",
			batch.LabelLengths,
		)
	}
}
func TestBatchToTensor(t *testing.T) {
	batch := &Batch{
		Features: NewFeatureMatrix(6, 2),

		BatchSize:  2,
		TimeSteps:  3,
		FeatureDim: 2,

		InputLengths: []int{3, 2},
		LabelLengths: []int{1, 1},
	}

	// sample 0
	batch.Features.Set(0, 0, 1)
	batch.Features.Set(0, 1, 2)

	batch.Features.Set(1, 0, 3)
	batch.Features.Set(1, 1, 4)

	batch.Features.Set(2, 0, 5)
	batch.Features.Set(2, 1, 6)

	// sample 1
	batch.Features.Set(3, 0, 7)
	batch.Features.Set(3, 1, 8)

	batch.Features.Set(4, 0, 9)
	batch.Features.Set(4, 1, 10)

	tensor, err := batch.ToTensor()
	if err != nil {
		t.Fatal(err)
	}

	t.Log(
		"Shape:",
		tensor.Batch,
		tensor.Time,
		tensor.Features,
	)

	value, err := tensor.Get(0, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	if value != 4 {
		t.Fatalf(
			"expected 4, got %f",
			value,
		)
	}

	// Padding must remain zero.
	value, err = tensor.Get(1, 2, 0)
	if err != nil {
		t.Fatal(err)
	}

	if value != 0 {
		t.Fatalf(
			"expected padding to be zero, got %f",
			value,
		)
	}
}
func TestLinearForward(t *testing.T) {
	linear := NewLinear(2, 3)

	if linear == nil {
		t.Fatal("expected linear")
	}

	// y0 = 1*x0 + 2*x1 + 1
	linear.SetWeight(0, 0, 1)
	linear.SetWeight(0, 1, 2)
	linear.SetBias(0, 1)

	// y1 = 3*x0 + 4*x1 + 2
	linear.SetWeight(1, 0, 3)
	linear.SetWeight(1, 1, 4)
	linear.SetBias(1, 2)

	// y2 = 5*x0 + 6*x1 + 3
	linear.SetWeight(2, 0, 5)
	linear.SetWeight(2, 1, 6)
	linear.SetBias(2, 3)

	output, err := linear.Forward([]float32{2, 3})
	if err != nil {
		t.Fatal(err)
	}

	t.Log("Output:", output)

	expected := []float32{
		9,
		20,
		31,
	}

	for i := range expected {
		if output[i] != expected[i] {
			t.Fatalf(
				"output[%d]: expected %f, got %f",
				i,
				expected[i],
				output[i],
			)
		}
	}
}
func TestLSTMCellForward(t *testing.T) {
	lstm := NewLSTMCell(2, 2)

	if lstm == nil {
		t.Fatal("expected lstm")
	}

	// فقط biasها را تنظیم می‌کنیم تا
	// رفتار Cell قابل پیش‌بینی باشد.

	// input gate
	lstm.SetBias(0, 0, 0)
	lstm.SetBias(0, 1, 0)

	// forget gate
	lstm.SetBias(1, 0, 0)
	lstm.SetBias(1, 1, 0)

	// candidate
	lstm.SetBias(2, 0, 1)
	lstm.SetBias(2, 1, 1)

	// output gate
	lstm.SetBias(3, 0, 0)
	lstm.SetBias(3, 1, 0)

	hidden := []float32{0, 0}
	cell := []float32{0, 0}

	nextHidden, nextCell, err := lstm.Forward(
		[]float32{0, 0},
		hidden,
		cell,
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Log("Hidden:", nextHidden)
	t.Log("Cell:", nextCell)

	if len(nextHidden) != 2 {
		t.Fatalf(
			"expected hidden size 2, got %d",
			len(nextHidden),
		)
	}

	if len(nextCell) != 2 {
		t.Fatalf(
			"expected cell size 2, got %d",
			len(nextCell),
		)
	}

	// i = sigmoid(0) = 0.5
	// g = tanh(1)
	// c = 0.5 * tanh(1)
	expectedCell := float32(0.5 * math.Tanh(1))

	for i := range nextCell {
		if math.Abs(
			float64(nextCell[i]-expectedCell),
		) > 1e-5 {
			t.Fatalf(
				"cell[%d]: expected %f, got %f",
				i,
				expectedCell,
				nextCell[i],
			)
		}
	}
}
func TestLSTMSequenceForward(t *testing.T) {
	input := NewTensor3D(2, 3, 2)

	if input == nil {
		t.Fatal("expected input tensor")
	}

	// Batch 0
	input.Set(0, 0, 0, 1)
	input.Set(0, 0, 1, 2)

	input.Set(0, 1, 0, 3)
	input.Set(0, 1, 1, 4)

	input.Set(0, 2, 0, 5)
	input.Set(0, 2, 1, 6)

	// Batch 1
	input.Set(1, 0, 0, 7)
	input.Set(1, 0, 1, 8)

	input.Set(1, 1, 0, 9)
	input.Set(1, 1, 1, 10)

	// timestep 2 intentionally remains padding.

	lstm := NewLSTMSequence(2, 4)

	if lstm == nil {
		t.Fatal("expected lstm sequence")
	}

	// Make candidate gate non-zero.
	// Gate order:
	// 0 = input
	// 1 = forget
	// 2 = candidate
	// 3 = output
	for hidden := 0; hidden < 4; hidden++ {
		if err := lstm.Cell.SetBias(2, hidden, 1); err != nil {
			t.Fatal(err)
		}
	}

	output, err := lstm.Forward(
		input,
		[]int{3, 2},
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Log(
		"Output shape:",
		output.Batch,
		output.Time,
		output.Features,
	)

	if output.Batch != 2 {
		t.Fatalf(
			"expected batch 2, got %d",
			output.Batch,
		)
	}

	if output.Time != 3 {
		t.Fatalf(
			"expected time 3, got %d",
			output.Time,
		)
	}

	if output.Features != 4 {
		t.Fatalf(
			"expected features 4, got %d",
			output.Features,
		)
	}

	// Batch 0 has 3 valid timesteps.
	for time := 0; time < 3; time++ {
		for feature := 0; feature < 4; feature++ {
			value, err := output.Get(
				0,
				time,
				feature,
			)

			if err != nil {
				t.Fatal(err)
			}

			if value == 0 {
				t.Fatalf(
					"expected non-zero output at batch=0 time=%d feature=%d",
					time,
					feature,
				)
			}
		}
	}

	// Batch 1 has only 2 valid timesteps.
	// Padding timestep must remain zero.
	for feature := 0; feature < 4; feature++ {
		value, err := output.Get(
			1,
			2,
			feature,
		)

		if err != nil {
			t.Fatal(err)
		}

		if value != 0 {
			t.Fatalf(
				"expected padding to remain zero, got %f",
				value,
			)
		}
	}
}
func TestBackwardLSTMSequenceForward(t *testing.T) {
	input := NewTensor3D(1, 3, 2)

	if input == nil {
		t.Fatal("expected input tensor")
	}

	input.Set(0, 0, 0, 1)
	input.Set(0, 0, 1, 2)

	input.Set(0, 1, 0, 3)
	input.Set(0, 1, 1, 4)

	input.Set(0, 2, 0, 5)
	input.Set(0, 2, 1, 6)

	lstm := NewBackwardLSTMSequence(2, 4)

	if lstm == nil {
		t.Fatal("expected backward lstm")
	}

	// Candidate gate produces deterministic non-zero output.
	for hidden := 0; hidden < 4; hidden++ {
		if err := lstm.Cell.SetBias(2, hidden, 1); err != nil {
			t.Fatal(err)
		}
	}

	output, err := lstm.Forward(
		input,
		[]int{3},
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Log(
		"Output shape:",
		output.Batch,
		output.Time,
		output.Features,
	)

	if output.Batch != 1 {
		t.Fatalf(
			"expected batch 1, got %d",
			output.Batch,
		)
	}

	if output.Time != 3 {
		t.Fatalf(
			"expected time 3, got %d",
			output.Time,
		)
	}

	if output.Features != 4 {
		t.Fatalf(
			"expected features 4, got %d",
			output.Features,
		)
	}

	for time := 0; time < 3; time++ {
		for feature := 0; feature < 4; feature++ {
			value, err := output.Get(
				0,
				time,
				feature,
			)

			if err != nil {
				t.Fatal(err)
			}

			if value == 0 {
				t.Fatalf(
					"expected non-zero output at time=%d feature=%d",
					time,
					feature,
				)
			}
		}
	}
}
func TestBiLSTMForward(t *testing.T) {
	input := NewTensor3D(1, 3, 2)

	if input == nil {
		t.Fatal("expected input tensor")
	}

	input.Set(0, 0, 0, 1)
	input.Set(0, 0, 1, 2)

	input.Set(0, 1, 0, 3)
	input.Set(0, 1, 1, 4)

	input.Set(0, 2, 0, 5)
	input.Set(0, 2, 1, 6)

	bilstm := NewBiLSTM(2, 4)

	if bilstm == nil {
		t.Fatal("expected bilstm")
	}

	// Make both directions deterministic and non-zero.
	for hidden := 0; hidden < 4; hidden++ {

		if err := bilstm.ForwardLSTM.Cell.SetBias(
			2,
			hidden,
			1,
		); err != nil {
			t.Fatal(err)
		}

		if err := bilstm.BackwardLSTM.Cell.SetBias(
			2,
			hidden,
			1,
		); err != nil {
			t.Fatal(err)
		}
	}

	output, err := bilstm.Forward(
		input,
		[]int{3},
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Log(
		"Output shape:",
		output.Batch,
		output.Time,
		output.Features,
	)

	if output.Batch != 1 {
		t.Fatalf(
			"expected batch 1, got %d",
			output.Batch,
		)
	}

	if output.Time != 3 {
		t.Fatalf(
			"expected time 3, got %d",
			output.Time,
		)
	}

	if output.Features != 8 {
		t.Fatalf(
			"expected features 8, got %d",
			output.Features,
		)
	}

	for time := 0; time < 3; time++ {
		for feature := 0; feature < 8; feature++ {
			value, err := output.Get(
				0,
				time,
				feature,
			)

			if err != nil {
				t.Fatal(err)
			}

			if value == 0 {
				t.Fatalf(
					"expected non-zero output at time=%d feature=%d",
					time,
					feature,
				)
			}
		}
	}
}
func TestLinearForwardTensor(t *testing.T) {
	input := NewTensor3D(1, 2, 2)

	if input == nil {
		t.Fatal("expected input")
	}

	input.Set(0, 0, 0, 1)
	input.Set(0, 0, 1, 2)

	input.Set(0, 1, 0, 3)
	input.Set(0, 1, 1, 4)

	linear := NewLinear(2, 3)

	if linear == nil {
		t.Fatal("expected linear")
	}

	// Output 0 = x0 + 2*x1 + 1
	linear.SetWeight(0, 0, 1)
	linear.SetWeight(0, 1, 2)
	linear.SetBias(0, 1)

	// Output 1 = 3*x0 + 4*x1 + 2
	linear.SetWeight(1, 0, 3)
	linear.SetWeight(1, 1, 4)
	linear.SetBias(1, 2)

	// Output 2 = 5*x0 + 6*x1 + 3
	linear.SetWeight(2, 0, 5)
	linear.SetWeight(2, 1, 6)
	linear.SetBias(2, 3)

	output, err := linear.ForwardTensor(input)

	if err != nil {
		t.Fatal(err)
	}

	t.Log(
		"Output shape:",
		output.Batch,
		output.Time,
		output.Features,
	)

	expected := []float32{
		6, 13, 20,
		12, 27, 42,
	}

	index := 0

	for time := 0; time < 2; time++ {
		for feature := 0; feature < 3; feature++ {
			value, err := output.Get(
				0,
				time,
				feature,
			)

			if err != nil {
				t.Fatal(err)
			}

			if value != expected[index] {
				t.Fatalf(
					"[%d,%d]: expected %f, got %f",
					time,
					feature,
					expected[index],
					value,
				)
			}

			index++
		}
	}
}
func TestCTCDecoder(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	decoder := NewCTCDecoder(vocabulary)

	if decoder == nil {
		t.Fatal("expected decoder")
	}

	// blank = 0
	//
	// hello:
	// h h e e blank l l o
	//
	// CTC collapse:
	// h e blank l o
	//
	// => "helo"
	//
	// برای "hello" باید بین دو l یک blank داشته باشیم.
	//
	// h h e blank l l blank l o
	// => hello

	logits := NewTensor3D(
		1,
		9,
		vocabulary.Size(),
	)

	h, _ := vocabulary.EncodeToken("h")
	e, _ := vocabulary.EncodeToken("e")
	l, _ := vocabulary.EncodeToken("l")
	o, _ := vocabulary.EncodeToken("o")

	sequence := []int{
		h,
		h,
		e,
		vocabulary.BlankID(),
		l,
		l,
		vocabulary.BlankID(),
		l,
		o,
	}

	for time, token := range sequence {
		if err := logits.Set(
			0,
			time,
			token,
			10,
		); err != nil {
			t.Fatal(err)
		}
	}

	results, err := decoder.Decode(
		logits,
		[]int{9},
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Log("Decoded:", results)

	if len(results) != 1 {
		t.Fatalf(
			"expected 1 result, got %d",
			len(results),
		)
	}

	if results[0] != "hello" {
		t.Fatalf(
			"expected hello, got %q",
			results[0],
		)
	}
}
func TestSTTModel(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		2,
		4,
		vocabulary,
	)

	if model == nil {
		t.Fatal("expected model")
	}

	input := NewTensor3D(
		1,
		6,
		2,
	)

	if input == nil {
		t.Fatal("expected input")
	}

	for time := 0; time < 6; time++ {
		input.Set(
			0,
			time,
			0,
			float32(time+1),
		)

		input.Set(
			0,
			time,
			1,
			float32(time+2),
		)
	}

	// برای اینکه مدل deterministic باشد،
	// فعلاً همه weightهای LSTM صفر هستند.
	//
	// Linear را هم طوری تنظیم می‌کنیم که
	// یک sequence مشخص تولید کند.

	h := vocabulary.Size()

	// Encoder candidate gate
	for hidden := 0; hidden < 4; hidden++ {
		model.Encoder.ForwardLSTM.Cell.SetBias(
			2,
			hidden,
			1,
		)

		model.Encoder.BackwardLSTM.Cell.SetBias(
			2,
			hidden,
			1,
		)
	}

	// همه logits را صفر نگه می‌داریم.
	// فقط برای تست pipeline مهم است که
	// مدل بدون panic کل مسیر را طی کند.

	logits, err := model.Forward(
		input,
		[]int{6},
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Log(
		"Logits shape:",
		logits.Batch,
		logits.Time,
		logits.Features,
	)

	if logits.Batch != 1 {
		t.Fatalf(
			"expected batch 1, got %d",
			logits.Batch,
		)
	}

	if logits.Time != 6 {
		t.Fatalf(
			"expected time 6, got %d",
			logits.Time,
		)
	}

	if logits.Features != h {
		t.Fatalf(
			"expected features %d, got %d",
			h,
			logits.Features,
		)
	}

	text, err := model.Transcribe(
		input,
		[]int{6},
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Log("Decoded:", text)

	if len(text) != 1 {
		t.Fatalf(
			"expected 1 result, got %d",
			len(text),
		)
	}
}
func TestCTCLoss(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	h, _ := vocabulary.EncodeToken("h")
	i, _ := vocabulary.EncodeToken("i")

	blank := vocabulary.BlankID()

	logits := NewTensor3D(
		1,
		4,
		vocabulary.Size(),
	)

	sequence := []int{
		h,
		h,
		blank,
		i,
	}

	for time, token := range sequence {
		if err := logits.Set(
			0,
			time,
			token,
			10,
		); err != nil {
			t.Fatal(err)
		}
	}

	loss, err := CTCLoss(
		logits,
		[]int{h, i},
		4,
		blank,
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Log("CTC Loss:", loss)

	if math.IsNaN(float64(loss)) {
		t.Fatal("loss is NaN")
	}

	if math.IsInf(float64(loss), 0) {
		t.Fatal("loss is Inf")
	}

	if loss <= 0 {
		t.Fatalf(
			"expected positive loss, got %f",
			loss,
		)
	}
}
func TestCTCLossBatch(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	h, _ := vocabulary.EncodeToken("h")
	i, _ := vocabulary.EncodeToken("i")

	blank := vocabulary.BlankID()

	logits := NewTensor3D(
		2,
		4,
		vocabulary.Size(),
	)

	sequence0 := []int{
		h,
		h,
		blank,
		i,
	}

	sequence1 := []int{
		i,
		i,
		blank,
		h,
	}

	for time, token := range sequence0 {
		if err := logits.Set(
			0,
			time,
			token,
			10,
		); err != nil {
			t.Fatal(err)
		}
	}

	for time, token := range sequence1 {
		if err := logits.Set(
			1,
			time,
			token,
			10,
		); err != nil {
			t.Fatal(err)
		}
	}

	hTarget := []int{h, i}
	iTarget := []int{i, h}

	targets := append(
		append([]int{}, hTarget...),
		iTarget...,
	)

	loss, err := CTCLossBatch(
		logits,
		targets,
		[]int{4, 4},
		[]int{2, 2},
		blank,
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Log("CTC Batch Loss:", loss)

	if math.IsNaN(float64(loss)) {
		t.Fatal("loss is NaN")
	}

	if math.IsInf(float64(loss), 0) {
		t.Fatal("loss is Inf")
	}

	if loss <= 0 {
		t.Fatalf(
			"expected positive loss, got %f",
			loss,
		)
	}
}
func TestCTCGradient(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	h, _ := vocabulary.EncodeToken("h")
	i, _ := vocabulary.EncodeToken("i")

	blank := vocabulary.BlankID()

	logits := NewTensor3D(
		1,
		4,
		vocabulary.Size(),
	)

	sequence := []int{
		h,
		h,
		blank,
		i,
	}

	for time, token := range sequence {
		if err := logits.Set(
			0,
			time,
			token,
			10,
		); err != nil {
			t.Fatal(err)
		}
	}

	gradient, err := CTCGradient(
		logits,
		[]int{h, i},
		4,
		blank,
	)

	if err != nil {
		t.Fatal(err)
	}

	if gradient.Batch != logits.Batch ||
		gradient.Time != logits.Time ||
		gradient.Features != logits.Features {
		t.Fatalf(
			"invalid gradient shape: %d %d %d",
			gradient.Batch,
			gradient.Time,
			gradient.Features,
		)
	}

	for time := 0; time < 4; time++ {
		var sum float64

		for token := 0; token < vocabulary.Size(); token++ {
			value, err := gradient.Get(
				0,
				time,
				token,
			)

			if err != nil {
				t.Fatal(err)
			}

			if math.IsNaN(float64(value)) {
				t.Fatalf(
					"gradient is NaN at time=%d token=%d",
					time,
					token,
				)
			}

			if math.IsInf(float64(value), 0) {
				t.Fatalf(
					"gradient is Inf at time=%d token=%d",
					time,
					token,
				)
			}

			sum += float64(value)
		}

		if math.Abs(sum) > 1e-4 {
			t.Fatalf(
				"gradient sum should be near zero at time=%d, got %f",
				time,
				sum,
			)
		}
	}

	t.Log("CTC Gradient shape:",
		gradient.Batch,
		gradient.Time,
		gradient.Features,
	)
}
func TestLinearBackward(t *testing.T) {
	linear := NewLinear(2, 3)

	_ = linear.SetWeight(0, 0, 1)
	_ = linear.SetWeight(0, 1, 2)

	_ = linear.SetWeight(1, 0, 3)
	_ = linear.SetWeight(1, 1, 4)

	_ = linear.SetWeight(2, 0, 5)
	_ = linear.SetWeight(2, 1, 6)

	input := []float32{
		10,
		20,
	}

	gradient := []float32{
		1,
		2,
		3,
	}

	inputGradient, parameterGradient, err :=
		linear.Backward(
			input,
			gradient,
		)

	if err != nil {
		t.Fatal(err)
	}

	t.Log("Input gradient:", inputGradient)
	t.Log("Weight gradient:", parameterGradient.Weights)
	t.Log("Bias gradient:", parameterGradient.Bias)

	expectedInputGradient := []float32{
		22,
		28,
	}

	for i, expected := range expectedInputGradient {
		if inputGradient[i] != expected {
			t.Fatalf(
				"input gradient[%d]: expected %f, got %f",
				i,
				expected,
				inputGradient[i],
			)
		}
	}

	expectedWeights := []float32{
		10, 20,
		20, 40,
		30, 60,
	}

	for i, expected := range expectedWeights {
		if parameterGradient.Weights[i] != expected {
			t.Fatalf(
				"weight gradient[%d]: expected %f, got %f",
				i,
				expected,
				parameterGradient.Weights[i],
			)
		}
	}

	expectedBias := []float32{
		1,
		2,
		3,
	}

	for i, expected := range expectedBias {
		if parameterGradient.Bias[i] != expected {
			t.Fatalf(
				"bias gradient[%d]: expected %f, got %f",
				i,
				expected,
				parameterGradient.Bias[i],
			)
		}
	}
}
func TestLinearBackwardTensor(t *testing.T) {
	linear := NewLinear(2, 3)
	_ = linear.SetWeight(0, 0, 1)
	_ = linear.SetWeight(0, 1, 2)
	_ = linear.SetWeight(1, 0, 3)
	_ = linear.SetWeight(1, 1, 4)
	_ = linear.SetWeight(2, 0, 5)
	_ = linear.SetWeight(2, 1, 6)
	input := NewTensor3D(1, 2, 2)
	_ = input.Set(0, 0, 0, 10)
	_ = input.Set(0, 0, 1, 20)
	_ = input.Set(0, 1, 0, 2)
	_ = input.Set(0, 1, 1, 3)
	gradient := NewTensor3D(1, 2, 3)
	_ = gradient.Set(0, 0, 0, 1)
	_ = gradient.Set(0, 0, 1, 2)
	_ = gradient.Set(0, 0, 2, 3)
	_ = gradient.Set(0, 1, 0, 4)
	_ = gradient.Set(0, 1, 1, 5)
	_ = gradient.Set(0, 1, 2, 6)
	inputGradient, parameterGradient, err :=
		linear.BackwardTensor(
			input,
			gradient,
		)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(
		"Input gradient:",
		inputGradient.Data,
	)
	t.Log(
		"Weight gradient:",
		parameterGradient.Weights,
	)
	t.Log(
		"Bias gradient:",
		parameterGradient.Bias,
	)
	expectedInput := []float32{
		22, 28,
		49, 64,
	}
	for i, expected := range expectedInput {
		if inputGradient.Data[i] != expected {
			t.Fatalf(
				"input gradient[%d]: expected %f, got %f",
				i,
				expected,
				inputGradient.Data[i],
			)
		}
	}
	expectedWeights := []float32{
		18, 32,
		30, 55,
		42, 78,
	}

	for i, expected := range expectedWeights {
		if parameterGradient.Weights[i] != expected {
			t.Fatalf(
				"weight gradient[%d]: expected %f, got %f",
				i,
				expected,
				parameterGradient.Weights[i],
			)
		}
	}

	expectedBias := []float32{
		5,
		7,
		9,
	}

	for i, expected := range expectedBias {
		if parameterGradient.Bias[i] != expected {
			t.Fatalf(
				"bias gradient[%d]: expected %f, got %f",
				i,
				expected,
				parameterGradient.Bias[i],
			)
		}
	}
}
func TestLSTMCellBackward(t *testing.T) {
	lstm := NewLSTMCell(2, 2)

	for gate := 0; gate < 4; gate++ {
		for hidden := 0; hidden < 2; hidden++ {
			for input := 0; input < 4; input++ {
				if err := lstm.SetWeight(
					gate,
					hidden,
					input,
					0.1,
				); err != nil {
					t.Fatal(err)
				}
			}

			if err := lstm.SetBias(
				gate,
				hidden,
				0.1,
			); err != nil {
				t.Fatal(err)
			}
		}
	}

	input := []float32{
		0.5,
		-0.25,
	}

	hidden := []float32{
		0.2,
		-0.1,
	}

	cell := []float32{
		0.3,
		-0.2,
	}

	nextHiddenGradient := []float32{
		1,
		2,
	}

	nextCellGradient := []float32{
		0.5,
		1,
	}

	result, err := lstm.Backward(
		input,
		hidden,
		cell,
		nextHiddenGradient,
		nextCellGradient,
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Log(
		"Input gradient:",
		result.InputGradient,
	)

	t.Log(
		"Hidden gradient:",
		result.HiddenGradient,
	)

	t.Log(
		"Cell gradient:",
		result.CellGradient,
	)

	for _, value := range result.InputGradient {
		if math.IsNaN(float64(value)) ||
			math.IsInf(float64(value), 0) {
			t.Fatal("invalid input gradient")
		}
	}

	for _, value := range result.HiddenGradient {
		if math.IsNaN(float64(value)) ||
			math.IsInf(float64(value), 0) {
			t.Fatal("invalid hidden gradient")
		}
	}

	for _, value := range result.CellGradient {
		if math.IsNaN(float64(value)) ||
			math.IsInf(float64(value), 0) {
			t.Fatal("invalid cell gradient")
		}
	}

	if len(result.Parameters.Weights) !=
		len(lstm.Weights) {
		t.Fatal("invalid weight gradient size")
	}

	if len(result.Parameters.Bias) !=
		len(lstm.Bias) {
		t.Fatal("invalid bias gradient size")
	}
}
func TestLSTMSequenceBackward(t *testing.T) {
	lstm := NewLSTMSequence(2, 2)

	for gate := 0; gate < 4; gate++ {
		for hidden := 0; hidden < 2; hidden++ {
			for input := 0; input < 4; input++ {
				if err := lstm.Cell.SetWeight(
					gate,
					hidden,
					input,
					0.1,
				); err != nil {
					t.Fatal(err)
				}
			}

			if err := lstm.Cell.SetBias(
				gate,
				hidden,
				0.1,
			); err != nil {
				t.Fatal(err)
			}
		}
	}

	input := NewTensor3D(1, 3, 2)

	_ = input.Set(0, 0, 0, 0.5)
	_ = input.Set(0, 0, 1, -0.25)

	_ = input.Set(0, 1, 0, 0.2)
	_ = input.Set(0, 1, 1, 0.1)

	_ = input.Set(0, 2, 0, -0.3)
	_ = input.Set(0, 2, 1, 0.4)

	output, err := lstm.Forward(
		input,
		[]int{3},
	)

	if err != nil {
		t.Fatal(err)
	}

	outputGradient := NewTensor3D(
		1,
		3,
		2,
	)

	for time := 0; time < 3; time++ {
		for feature := 0; feature < 2; feature++ {
			_ = outputGradient.Set(
				0,
				time,
				feature,
				1,
			)
		}
	}

	gradient, err := lstm.Backward(
		input,
		[]int{3},
		outputGradient,
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Log(
		"Output shape:",
		output.Batch,
		output.Time,
		output.Features,
	)

	t.Log(
		"Input gradient:",
		gradient.Input.Data,
	)

	for _, value := range gradient.Input.Data {
		if math.IsNaN(float64(value)) ||
			math.IsInf(float64(value), 0) {
			t.Fatal("invalid input gradient")
		}
	}

	if len(gradient.Parameters.Weights) !=
		len(lstm.Cell.Weights) {
		t.Fatal("invalid weight gradient size")
	}

	if len(gradient.Parameters.Bias) !=
		len(lstm.Cell.Bias) {
		t.Fatal("invalid bias gradient size")
	}
}
func TestBiLSTMBackward(t *testing.T) {
	bilstm := NewBiLSTM(2, 2)

	for _, cell := range []*LSTMCell{
		bilstm.ForwardLSTM.Cell,
		bilstm.BackwardLSTM.Cell,
	} {
		for gate := 0; gate < 4; gate++ {
			for hidden := 0; hidden < 2; hidden++ {
				for input := 0; input < 4; input++ {
					if err := cell.SetWeight(
						gate,
						hidden,
						input,
						0.1,
					); err != nil {
						t.Fatal(err)
					}
				}

				if err := cell.SetBias(
					gate,
					hidden,
					0.1,
				); err != nil {
					t.Fatal(err)
				}
			}
		}
	}

	input := NewTensor3D(1, 3, 2)

	if err := input.Set(0, 0, 0, 0.5); err != nil {
		t.Fatal(err)
	}

	if err := input.Set(0, 0, 1, -0.25); err != nil {
		t.Fatal(err)
	}

	if err := input.Set(0, 1, 0, 0.2); err != nil {
		t.Fatal(err)
	}

	if err := input.Set(0, 1, 1, 0.1); err != nil {
		t.Fatal(err)
	}

	if err := input.Set(0, 2, 0, -0.3); err != nil {
		t.Fatal(err)
	}

	if err := input.Set(0, 2, 1, 0.4); err != nil {
		t.Fatal(err)
	}

	output, err := bilstm.Forward(
		input,
		[]int{3},
	)

	if err != nil {
		t.Fatal(err)
	}

	outputGradient := NewTensor3D(
		1,
		3,
		4,
	)

	for time := 0; time < 3; time++ {
		for feature := 0; feature < 4; feature++ {
			if err := outputGradient.Set(
				0,
				time,
				feature,
				1,
			); err != nil {
				t.Fatal(err)
			}
		}
	}

	// IMPORTANT:
	// Backward must be called on the whole BiLSTM.
	result, err := bilstm.Backward(
		input,
		[]int{3},
		outputGradient,
	)

	if err != nil {
		t.Fatal(err)
	}

	if result == nil {
		t.Fatal("nil bilstm gradient")
	}

	if result.Input == nil {
		t.Fatal("nil input gradient")
	}

	t.Log(
		"Output shape:",
		output.Batch,
		output.Time,
		output.Features,
	)

	t.Log(
		"Input gradient:",
		result.Input.Data,
	)

	for _, value := range result.Input.Data {
		if math.IsNaN(float64(value)) ||
			math.IsInf(float64(value), 0) {
			t.Fatal("invalid input gradient")
		}
	}

	if result.ForwardParameters == nil {
		t.Fatal("nil forward parameters")
	}

	if result.BackwardParameters == nil {
		t.Fatal("nil backward parameters")
	}

	if len(result.ForwardParameters.Weights) !=
		len(bilstm.ForwardLSTM.Cell.Weights) {
		t.Fatal("invalid forward weight gradient")
	}

	if len(result.ForwardParameters.Bias) !=
		len(bilstm.ForwardLSTM.Cell.Bias) {
		t.Fatal("invalid forward bias gradient")
	}

	if len(result.BackwardParameters.Weights) !=
		len(bilstm.BackwardLSTM.Cell.Weights) {
		t.Fatal("invalid backward weight gradient")
	}

	if len(result.BackwardParameters.Bias) !=
		len(bilstm.BackwardLSTM.Cell.Bias) {
		t.Fatal("invalid backward bias gradient")
	}
}
func TestSTTModelBackward(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		4,
		vocabulary,
	)
	input := NewTensor3D(
		1,
		6,
		80,
	)

	for time := 0; time < 6; time++ {
		for feature := 0; feature < 80; feature++ {
			if err := input.Set(
				0,
				time,
				feature,
				0.01,
			); err != nil {
				t.Fatal(err)
			}
		}
	}

	targets := []int{
		8,
		5,
		12,
		12,
		15,
	}

	result, err := model.Backward(
		input,
		[]int{6},
		targets,
		[]int{5},
	)

	if err != nil {
		t.Fatal(err)
	}

	if result == nil {
		t.Fatal("nil model gradient")
	}

	if result.Encoder == nil {
		t.Fatal("nil encoder gradient")
	}

	if result.Decoder == nil {
		t.Fatal("nil decoder gradient")
	}

	if result.Encoder.Input == nil {
		t.Fatal("nil encoder input gradient")
	}

	t.Log(
		"Input gradient shape:",
		result.Encoder.Input.Batch,
		result.Encoder.Input.Time,
		result.Encoder.Input.Features,
	)

	t.Log(
		"Decoder weight gradient:",
		result.Decoder.Weights,
	)

	for _, value := range result.Encoder.Input.Data {
		if math.IsNaN(float64(value)) ||
			math.IsInf(float64(value), 0) {
			t.Fatal("invalid encoder gradient")
		}
	}

	for _, value := range result.Decoder.Weights {
		if math.IsNaN(float64(value)) ||
			math.IsInf(float64(value), 0) {
			t.Fatal("invalid decoder gradient")
		}
	}
}
func TestSGD(t *testing.T) {
	optimizer, err := NewSGD(0.1)
	if err != nil {
		t.Fatal(err)
	}
	parameters := []float32{
		1,
		2,
		3,
	}
	gradients := []float32{
		0.5,
		1,
		2,
	}
	opterr := optimizer.Step([]*Parameter{
		{
			Name:      "test",
			Values:    parameters,
			Gradients: gradients,
		},
	})
	if opterr != nil {
		t.Fatal(opterr)
	}
	expected := []float32{
		0.95,
		1.9,
		2.8,
	}
	for i := range expected {
		if math.Abs(
			float64(parameters[i]-expected[i]),
		) > 1e-6 {
			t.Fatalf(
				"parameter[%d]: got %f expected %f",
				i,
				parameters[i],
				expected[i],
			)
		}
	}
	t.Logf("Updated parameters: %v", parameters)
}
func TestWeightInitialization(t *testing.T) {

	rng := rand.New(
		rand.NewSource(42),
	)

	linear := NewLinear(4, 3)

	InitializeLinear(
		linear,
		rng,
	)

	allZero := true

	for _, value := range linear.Weights {
		if value != 0 {
			allZero = false
			break
		}
	}

	if allZero {
		t.Fatal("linear weights are all zero")
	}

	for _, value := range linear.Bias {
		if value != 0 {
			t.Fatal("linear bias must start at zero")
		}
	}

	lstm := NewLSTMCell(4, 3)

	InitializeLSTMCell(
		lstm,
		rng,
	)

	allZero = true

	for _, value := range lstm.Weights {
		if value != 0 {
			allZero = false
			break
		}
	}

	if allZero {
		t.Fatal("lstm weights are all zero")
	}

	for _, value := range lstm.Bias {
		if value != 0 {
			t.Fatal("lstm bias must start at zero")
		}
	}

	t.Log("Linear and LSTM weights initialized")
}
func TestSTTModelInitialization(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		4,
		vocabulary,
	)
	allZero := true

	for _, value := range model.Decoder.Weights {
		if value != 0 {
			allZero = false
			break
		}
	}

	if allZero {
		t.Fatal("decoder weights are all zero")
	}

	allZero = true

	for _, value := range model.Encoder.ForwardLSTM.Cell.Weights {
		if value != 0 {
			allZero = false
			break
		}
	}

	if allZero {
		t.Fatal("forward LSTM weights are all zero")
	}

	allZero = true

	for _, value := range model.Encoder.BackwardLSTM.Cell.Weights {
		if value != 0 {
			allZero = false
			break
		}
	}

	if allZero {
		t.Fatal("backward LSTM weights are all zero")
	}

	t.Log("STT model initialized successfully")
}
func TestSTTModelTrainStep(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		4,
		vocabulary,
	)

	if model == nil {
		t.Fatal("model is nil")
	}

	input := NewTensor3D(
		1,
		6,
		80,
	)

	for time := 0; time < 6; time++ {
		for feature := 0; feature < 80; feature++ {

			value := float32(
				time+feature+1,
			) * 0.001

			if err := input.Set(
				0,
				time,
				feature,
				value,
			); err != nil {
				t.Fatal(err)
			}
		}
	}

	targets := []int{
		8,
		5,
		12,
	}

	targetLengths := []int{
		3,
	}

	optimizer, err := NewSGD(
		0.01,
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := model.TrainStep(
		input,
		[]int{6},
		targets,
		targetLengths,
		optimizer,
	)

	if err != nil {
		t.Fatal(err)
	}

	if result == nil {
		t.Fatal("training result is nil")
	}

	if math.IsNaN(float64(result.Loss)) ||
		math.IsInf(float64(result.Loss), 0) {
		t.Fatalf(
			"invalid loss: %f",
			result.Loss,
		)
	}

	if result.Loss <= 0 {
		t.Fatalf(
			"invalid loss: %f",
			result.Loss,
		)
	}

	t.Logf(
		"Training loss: %f",
		result.Loss,
	)
}
func TestSTTModelTraining(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		4,
		vocabulary,
	)

	if model == nil {
		t.Fatal("model is nil")
	}

	input := NewTensor3D(
		1,
		12,
		80,
	)

	for time := 0; time < 12; time++ {
		for feature := 0; feature < 80; feature++ {

			value := float32(
				(time+1)*(feature+1),
			) * 0.0001

			if err := input.Set(
				0,
				time,
				feature,
				value,
			); err != nil {
				t.Fatal(err)
			}
		}
	}

	targets := []int{
		8,
		5,
		12,
	}

	optimizer, err := NewSGD(
		0.01,
	)
	if err != nil {
		t.Fatal(err)
	}

	var firstLoss float32
	var lastLoss float32

	for step := 0; step < 20; step++ {

		result, err := model.TrainStep(
			input,
			[]int{12},
			targets,
			[]int{3},
			optimizer,
		)

		if err != nil {
			t.Fatal(err)
		}

		if result == nil {
			t.Fatal("training result is nil")
		}

		if math.IsNaN(float64(result.Loss)) ||
			math.IsInf(float64(result.Loss), 0) {
			t.Fatalf(
				"invalid loss at step %d: %f",
				step,
				result.Loss,
			)
		}

		if step == 0 {
			firstLoss = result.Loss
		}

		lastLoss = result.Loss

		t.Logf(
			"step=%d loss=%f",
			step,
			result.Loss,
		)
	}

	if lastLoss >= firstLoss {
		t.Fatalf(
			"loss did not decrease: first=%f last=%f",
			firstLoss,
			lastLoss,
		)
	}
}
func TestTrainer(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		4,
		vocabulary,
	)

	optimizer, err := NewSGD(
		0.01,
	)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	features := NewFeatureMatrix(
		12,
		80,
	)

	for row := 0; row < 12; row++ {
		for col := 0; col < 80; col++ {
			features.Set(
				row,
				col,
				float32(row+col+1)*0.0001,
			)
		}
	}

	err = dataset.Add(
		TrainingSample{
			Features: features,
			Labels: []int{
				8,
				5,
				12,
			},
			Text: "hel",
		},
	)

	if err != nil {
		t.Fatal(err)
	}

	batch, err := CreateBatch(
		dataset,
		0,
		1,
	)

	if err != nil {
		t.Fatal(err)
	}

	loss, err := trainer.TrainBatch(
		batch,
		batch.Labels,
	)

	if err != nil {
		t.Fatal(err)
	}

	if math.IsNaN(float64(loss)) ||
		math.IsInf(float64(loss), 0) {
		t.Fatalf(
			"invalid loss: %f",
			loss,
		)
	}

	if loss <= 0 {
		t.Fatalf(
			"invalid loss: %f",
			loss,
		)
	}

	t.Logf(
		"Trainer loss: %f",
		loss,
	)
}
func TestTrainerTrainEpoch(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(80, 8, vocabulary)
	if model == nil {
		t.Fatal("model is nil")
	}

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(model, optimizer)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	for i := 0; i < 4; i++ {
		features := NewFeatureMatrix(20, 80)

		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				features.Set(
					row,
					col,
					float32((row+1)*(col+1))*0.0001,
				)
			}
		}
		tokenizer, err := NewTokenizer(vocabulary)
		if err != nil {
			t.Fatal(err)
		}
		labels, err := tokenizer.Encode("hel")
		err = dataset.Add(TrainingSample{
			Features: features,
			Labels:   labels,
			Text:     "hel",
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	loader, err := NewDataLoader(
		dataset,
		2,
		true,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatal(err)
	}

	loss1, err := trainer.TrainEpoch(loader)
	if err != nil {
		t.Fatal(err)
	}

	loss2, err := trainer.TrainEpoch(loader)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("epoch 1 loss: %f", loss1)
	t.Logf("epoch 2 loss: %f", loss2)

	if loss2 >= loss1 {
		t.Fatalf(
			"loss did not decrease: first=%f second=%f",
			loss1,
			loss2,
		)
	}
}
func TestTrainerEpoch(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		4,
		vocabulary,
	)

	optimizer, err := NewSGD(0.01)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	for sampleIndex := 0; sampleIndex < 4; sampleIndex++ {

		features := NewFeatureMatrix(
			12,
			80,
		)

		for row := 0; row < 12; row++ {
			for col := 0; col < 80; col++ {

				value := float32(
					(sampleIndex+1)*
						(row+1)*
						(col+1),
				) * 0.00001

				features.Set(
					row,
					col,
					value,
				)
			}
		}

		err := dataset.Add(
			TrainingSample{
				Features: features,
				Labels: []int{
					8,
					5,
					12,
				},
				Text: "hel",
			},
		)

		if err != nil {
			t.Fatal(err)
		}
	}

	loader, err := NewDataLoader(
		dataset,
		2,
		true,
		trainer.RNG,
	)
	if err != nil {
		t.Fatal(err)
	}

	loss, err := trainer.TrainEpoch(loader)

	if err != nil {
		t.Fatal(err)
	}

	if math.IsNaN(float64(loss)) ||
		math.IsInf(float64(loss), 0) {
		t.Fatalf(
			"invalid epoch loss: %f",
			loss,
		)
	}

	if loss <= 0 {
		t.Fatalf(
			"invalid epoch loss: %f",
			loss,
		)
	}

	t.Logf(
		"Epoch loss: %f",
		loss,
	)
}
func TestTrainerMultiEpoch(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		4,
		vocabulary,
	)

	optimizer, err := NewSGD(0.01)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	for sampleIndex := 0; sampleIndex < 4; sampleIndex++ {

		features := NewFeatureMatrix(
			12,
			80,
		)

		for row := 0; row < 12; row++ {
			for col := 0; col < 80; col++ {

				value := float32(
					(sampleIndex+1)*
						(row+1)*
						(col+1),
				) * 0.00001

				features.Set(
					row,
					col,
					value,
				)
			}
		}

		if err := dataset.Add(
			TrainingSample{
				Features: features,
				Labels: []int{
					8,
					5,
					12,
				},
				Text: "hel",
			},
		); err != nil {
			t.Fatal(err)
		}
	}

	var firstLoss float32
	var lastLoss float32
	loader, err := NewDataLoader(
		dataset,
		2,
		true,
		trainer.RNG,
	)
	if err != nil {
		t.Fatal(err)
	}
	for epoch := 0; epoch < 10; epoch++ {
		loss, err := trainer.TrainEpoch(loader)

		if err != nil {
			t.Fatal(err)
		}

		if math.IsNaN(float64(loss)) ||
			math.IsInf(float64(loss), 0) {
			t.Fatalf(
				"invalid loss at epoch %d: %f",
				epoch,
				loss,
			)
		}

		if epoch == 0 {
			firstLoss = loss
		}

		lastLoss = loss

		t.Logf(
			"epoch=%d loss=%f",
			epoch,
			loss,
		)
	}

	if lastLoss >= firstLoss {
		t.Fatalf(
			"loss did not decrease: first=%f last=%f",
			firstLoss,
			lastLoss,
		)
	}
}
func TestTrainingTranscribeIntegration(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		4,
		vocabulary,
	)

	if model == nil {
		t.Fatal("model is nil")
	}

	optimizer, err := NewSGD(0.01)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	for sampleIndex := 0; sampleIndex < 4; sampleIndex++ {

		features := NewFeatureMatrix(
			12,
			80,
		)

		for row := 0; row < 12; row++ {
			for col := 0; col < 80; col++ {

				value := float32(
					(sampleIndex+1)*
						(row+1)*
						(col+1),
				) * 0.00001

				features.Set(
					row,
					col,
					value,
				)
			}
		}

		if err := dataset.Add(
			TrainingSample{
				Features: features,
				Labels: []int{
					8,
					5,
					12,
				},
				Text: "hel",
			},
		); err != nil {
			t.Fatal(err)
		}
	}

	losses, err := trainer.Train(
		dataset,
		TrainingConfig{
			Epochs:    100,
			BatchSize: 2,
		},
	)

	if err != nil {
		t.Fatal(err)
	}

	if len(losses) != 100 {
		t.Fatalf(
			"loss count: got %d expected 10",
			len(losses),
		)
	}

	t.Logf(
		"Initial loss: %f",
		losses[0],
	)

	t.Logf(
		"Final loss: %f",
		losses[len(losses)-1],
	)

	if losses[len(losses)-1] >= losses[0] {
		t.Fatalf(
			"training did not reduce loss: first=%f last=%f",
			losses[0],
			losses[len(losses)-1],
		)
	}

	// Use one trained sample for inference.
	sample, err := dataset.Get(0)
	if err != nil {
		t.Fatal(err)
	}

	input := NewTensor3D(
		1,
		sample.Features.Rows,
		sample.Features.Cols,
	)

	for time := 0; time < sample.Features.Rows; time++ {
		for feature := 0; feature < sample.Features.Cols; feature++ {

			value := sample.Features.Get(
				time,
				feature,
			)

			if err := input.Set(
				0,
				time,
				feature,
				value,
			); err != nil {
				t.Fatal(err)
			}
		}
	}
	encoderOutput, err := model.Encoder.Forward(
		input,
		[]int{sample.Features.Rows},
	)
	if err != nil {
		t.Fatal(err)
	}

	logits, err := model.Decoder.ForwardTensor(
		encoderOutput,
	)
	if err != nil {
		t.Fatal(err)
	}
	for time := 0; time < logits.Time; time++ {
		bestID := 0
		bestValue := logits.Data[time*logits.Features]

		for id := 1; id < logits.Features; id++ {
			value := logits.Data[time*logits.Features+id]

			if value > bestValue {
				bestValue = value
				bestID = id
			}
		}

		token, err := model.CTC.Vocabulary.DecodeID(bestID)
		if err != nil {
			t.Logf(
				"time=%d id=%d logit=%f decode_error=%v",
				time,
				bestID,
				bestValue,
				err,
			)
			continue
		}

		t.Logf(
			"time=%d id=%d token=%q logit=%f",
			time,
			bestID,
			token,
			bestValue,
		)
	}
	decoded, err := model.CTC.Decode(
		logits,
		[]int{sample.Features.Rows},
	)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf(
		"CTC decoded: %q",
		decoded,
	)
	hasNonZeroLogit := false

	for _, value := range logits.Data {
		if math.Abs(float64(value)) > 1e-8 {
			hasNonZeroLogit = true
			break
		}
	}

	if !hasNonZeroLogit {
		t.Fatal("all logits are zero after training")
	}

	t.Logf(
		"Logits shape: %d %d %d",
		logits.Batch,
		logits.Time,
		logits.Features,
	)
	result, err := model.Transcribe(
		input,
		[]int{sample.Features.Rows},
	)

	if err != nil {
		t.Fatal(err)
	}

	if len(result) != 1 {
		t.Fatalf(
			"transcription result count: got %d expected 1",
			len(result),
		)
	}

	t.Logf(
		"Transcription after training: %q",
		result[0],
	)
}
func TestCTCGradientNumerical(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	logits := NewTensor3D(
		1,
		4,
		vocabulary.Size(),
	)

	for i := range logits.Data {
		logits.Data[i] = float32(
			0.01 * float64(i+1),
		)
	}

	target := []int{
		8,
		5,
	}

	analytic, err := CTCGradient(
		logits,
		target,
		4,
		vocabulary.BlankID(),
	)

	if err != nil {
		t.Fatal(err)
	}

	epsilon := float32(1e-3)

	maxDifference := float32(0)

	for index := 0; index < len(logits.Data); index++ {

		original := logits.Data[index]

		logits.Data[index] = original + epsilon

		lossPlus, err := CTCLoss(
			logits,
			target,
			4,
			vocabulary.BlankID(),
		)

		if err != nil {
			t.Fatal(err)
		}

		logits.Data[index] = original - epsilon

		lossMinus, err := CTCLoss(
			logits,
			target,
			4,
			vocabulary.BlankID(),
		)

		if err != nil {
			t.Fatal(err)
		}

		logits.Data[index] = original

		numerical :=
			(lossPlus - lossMinus) /
				(2 * epsilon)

		difference := float32(
			math.Abs(
				float64(
					numerical -
						analytic.Data[index],
				),
			),
		)

		if difference > maxDifference {
			maxDifference = difference
		}
	}

	t.Logf(
		"Maximum CTC gradient difference: %f",
		maxDifference,
	)

	if maxDifference > 1e-2 {
		t.Fatalf(
			"CTC gradient mismatch: %f",
			maxDifference,
		)
	}
}
func TestSTTModelSingleSampleOverfit(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	if model == nil {
		t.Fatal("model is nil")
	}

	optimizer, err := NewSGD(0.05)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	features := NewFeatureMatrix(
		20,
		80,
	)

	for row := 0; row < 20; row++ {
		for col := 0; col < 80; col++ {

			value := float32(
				(row+1)*(col+1),
			) * 0.0001

			features.Set(
				row,
				col,
				value,
			)
		}
	}

	err = dataset.Add(
		TrainingSample{
			Features: features,
			Labels: []int{
				8,
				5,
				12,
			},
			Text: "hel",
		},
	)

	if err != nil {
		t.Fatal(err)
	}

	config := TrainingConfig{
		Epochs:    3000,
		BatchSize: 1,
	}

	losses, err := trainer.Train(
		dataset,
		config,
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Logf(
		"Initial loss: %f",
		losses[0],
	)

	t.Logf(
		"Final loss: %f",
		losses[len(losses)-1],
	)
	for i, loss := range losses {
		if i%100 == 0 || i == len(losses)-1 {
			t.Logf(
				"epoch=%d loss=%f",
				i,
				loss,
			)
		}
	}
	sample, err := dataset.Get(0)
	if err != nil {
		t.Fatal(err)
	}

	input := NewTensor3D(
		1,
		sample.Features.Rows,
		sample.Features.Cols,
	)

	for time := 0; time < sample.Features.Rows; time++ {
		for feature := 0; feature < sample.Features.Cols; feature++ {

			value := sample.Features.Get(
				time,
				feature,
			)

			if err := input.Set(
				0,
				time,
				feature,
				value,
			); err != nil {
				t.Fatal(err)
			}
		}
	}

	result, err := model.Transcribe(
		input,
		[]int{sample.Features.Rows},
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Logf(
		"Transcription: %q",
		result[0],
	)
}
func TestSTTModelSingleSampleOverfitOptimizer(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	if model == nil {
		t.Fatal("model is nil")
	}

	optimizer, err := NewAdam(0.001)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	features := NewFeatureMatrix(
		20,
		80,
	)

	for row := 0; row < 20; row++ {
		for col := 0; col < 80; col++ {

			value := float32(
				(row+1)*(col+1),
			) * 0.0001

			features.Set(
				row,
				col,
				value,
			)
		}
	}

	err = dataset.Add(
		TrainingSample{
			Features: features,
			Labels: []int{
				8,
				5,
				12,
			},
			Text: "hel",
		},
	)

	if err != nil {
		t.Fatal(err)
	}

	config := TrainingConfig{
		Epochs:    500,
		BatchSize: 1,
	}

	losses, err := trainer.Train(
		dataset,
		config,
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Logf(
		"Initial loss: %f",
		losses[0],
	)

	t.Logf(
		"Final loss: %f",
		losses[len(losses)-1],
	)
	for i, loss := range losses {
		if i%100 == 0 || i == len(losses)-1 {
			t.Logf(
				"epoch=%d loss=%f",
				i,
				loss,
			)
		}
	}
	sample, err := dataset.Get(0)
	if err != nil {
		t.Fatal(err)
	}

	input := NewTensor3D(
		1,
		sample.Features.Rows,
		sample.Features.Cols,
	)

	for time := 0; time < sample.Features.Rows; time++ {
		for feature := 0; feature < sample.Features.Cols; feature++ {

			value := sample.Features.Get(
				time,
				feature,
			)

			if err := input.Set(
				0,
				time,
				feature,
				value,
			); err != nil {
				t.Fatal(err)
			}
		}
	}

	result, err := model.Transcribe(
		input,
		[]int{sample.Features.Rows},
	)

	if err != nil {
		t.Fatal(err)
	}

	t.Logf(
		"Transcription: %q",
		result[0],
	)
}
func TestAdam(t *testing.T) {
	optimizer, err := NewAdam(0.001)
	if err != nil {
		t.Fatal(err)
	}

	parameters := []float32{
		1.0,
		2.0,
		3.0,
	}

	gradients := []float32{
		0.5,
		1.0,
		1.5,
	}

	parameter := &Parameter{
		Name:      "test",
		Values:    parameters,
		Gradients: gradients,
	}

	for i := 0; i < 10; i++ {
		if err := optimizer.Step(
			[]*Parameter{parameter},
		); err != nil {
			t.Fatal(err)
		}
	}

	t.Logf(
		"Adam parameters: %v",
		parameter.Values,
	)

	if optimizer.StepCount != 10 {
		t.Fatalf(
			"expected step count 10, got %d",
			optimizer.StepCount,
		)
	}
}
func TestAdamLearningRateSweep(t *testing.T) {
	learningRates := []float32{
		0.0005,
		0.001,
		0.002,
		0.005,
		0.01,
	}

	for _, learningRate := range learningRates {

		vocabulary, err := NewEnglishVocabulary()
		if err != nil {
			t.Fatal(err)
		}

		model := NewSTTModel(
			80,
			8,
			vocabulary,
		)

		if model == nil {
			t.Fatal("expected model")
		}

		optimizer, err := NewAdam(learningRate)
		if err != nil {
			t.Fatal(err)
		}

		trainer, err := NewTrainer(
			model,
			optimizer,
		)
		if err != nil {
			t.Fatal(err)
		}
		t.Log(trainer)
		input := NewTensor3D(
			1,
			20,
			80,
		)

		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				input.Set(
					0,
					row,
					col,
					float32(
						(row+1)*(col+1),
					)*0.0001,
				)
			}
		}

		targets := []int{
			8,
			5,
			12,
		}

		targetLengths := []int{3}
		inputLengths := []int{20}

		initialResult, err := model.TrainStep(
			input,
			inputLengths,
			targets,
			targetLengths,
			optimizer,
		)
		if err != nil {
			t.Fatal(err)
		}

		losses := make([]float32, 500)
		losses[0] = initialResult.Loss

		for epoch := 1; epoch < 500; epoch++ {
			result, err := model.TrainStep(
				input,
				inputLengths,
				targets,
				targetLengths,
				optimizer,
			)
			if err != nil {
				t.Fatal(err)
			}

			losses[epoch] = result.Loss
		}

		t.Logf(
			"lr=%.4f initial=%.6f final=%.6f",
			learningRate,
			losses[0],
			losses[len(losses)-1],
		)

		logits, err := model.Forward(
			input,
			inputLengths,
		)
		if err != nil {
			t.Fatal(err)
		}

		texts, err := model.CTC.Decode(
			logits,
			inputLengths,
		)
		if err != nil {
			t.Fatal(err)
		}

		t.Logf(
			"lr=%.4f transcription=%q",
			learningRate,
			texts[0],
		)
	}
}
func TestAdamLearningRateSweep2(t *testing.T) {
	learningRates := []float32{
		0.0075,
		0.01,
		0.0125,
		0.015,
		0.02,
	}

	for _, learningRate := range learningRates {

		vocabulary, err := NewEnglishVocabulary()
		if err != nil {
			t.Fatal(err)
		}

		model := NewSTTModel(
			80,
			8,
			vocabulary,
		)

		if model == nil {
			t.Fatal("expected model")
		}

		optimizer, err := NewAdam(learningRate)
		if err != nil {
			t.Fatal(err)
		}

		trainer, err := NewTrainer(
			model,
			optimizer,
		)
		if err != nil {
			t.Fatal(err)
		}
		t.Log(trainer)
		input := NewTensor3D(
			1,
			20,
			80,
		)

		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				input.Set(
					0,
					row,
					col,
					float32(
						(row+1)*(col+1),
					)*0.0001,
				)
			}
		}

		targets := []int{
			8,
			5,
			12,
		}

		targetLengths := []int{3}
		inputLengths := []int{20}

		initialResult, err := model.TrainStep(
			input,
			inputLengths,
			targets,
			targetLengths,
			optimizer,
		)
		if err != nil {
			t.Fatal(err)
		}

		losses := make([]float32, 500)
		losses[0] = initialResult.Loss

		for epoch := 1; epoch < 500; epoch++ {
			result, err := model.TrainStep(
				input,
				inputLengths,
				targets,
				targetLengths,
				optimizer,
			)
			if err != nil {
				t.Fatal(err)
			}

			losses[epoch] = result.Loss
		}

		t.Logf(
			"lr=%.4f initial=%.6f final=%.6f",
			learningRate,
			losses[0],
			losses[len(losses)-1],
		)

		logits, err := model.Forward(
			input,
			inputLengths,
		)
		if err != nil {
			t.Fatal(err)
		}

		texts, err := model.CTC.Decode(
			logits,
			inputLengths,
		)
		if err != nil {
			t.Fatal(err)
		}

		t.Logf(
			"lr=%.4f transcription=%q",
			learningRate,
			texts[0],
		)
	}
}
func TestAdamMultiSampleTraining(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	texts := []string{
		"hel",
		"hel",
		"hel",
		"hel",
	}

	for i, text := range texts {
		features := NewFeatureMatrix(20, 80)
		for row := range 20 {
			for col := range 80 {
				value := float32(
					(row+1)*(col+1),
				) * 0.0001

				// کمی variation بین sampleها
				value += float32(i) * 0.00001

				features.Set(
					row,
					col,
					value,
				)
			}
		}
		tokenizer, err := NewTokenizer(vocabulary)
		if err != nil {
			t.Fatal(err)
		}
		labels, err := tokenizer.Encode(text)
		if err != nil {
			t.Fatal(err)
		}

		err = dataset.Add(TrainingSample{
			Features: features,
			Labels:   labels,
			Text:     text,
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	initialBatch, err := CreateBatch(
		dataset,
		0,
		4,
	)
	if err != nil {
		t.Fatal(err)
	}

	initialLoss, err := trainer.TrainBatch(
		initialBatch,
		initialBatch.Labels,
	)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf(
		"Initial loss: %.6f",
		initialLoss,
	)
	loader, err := NewDataLoader(
		dataset,
		4,
		true,
		trainer.RNG,
	)
	if err != nil {
		t.Fatal(err)
	}
	for epoch := 0; epoch < 500; epoch++ {
		loss, err := trainer.TrainEpoch(loader)
		if err != nil {
			t.Fatal(err)
		}
		if epoch%100 == 0 ||
			epoch == 499 {
			t.Logf(
				"epoch=%d loss=%.6f",
				epoch,
				loss,
			)
		}
	}
}
func TestSTTModelSaveLoad(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	if model == nil {
		t.Fatal("expected model")
	}

	input := NewTensor3D(
		1,
		20,
		80,
	)

	for row := 0; row < 20; row++ {
		for col := 0; col < 80; col++ {
			input.Set(
				0,
				row,
				col,
				float32(
					(row+1)*(col+1),
				)*0.0001,
			)
		}
	}

	inputLengths := []int{20}

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	targets := []int{
		8,
		5,
		12,
	}

	targetLengths := []int{3}

	for i := 0; i < 500; i++ {
		_, err := model.TrainStep(
			input,
			inputLengths,
			targets,
			targetLengths,
			optimizer,
		)

		if err != nil {
			t.Fatal(err)
		}
	}

	before, err := model.Transcribe(
		input,
		inputLengths,
	)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf(
		"Before save: %q",
		before,
	)

	path := t.TempDir() + "/stt_model.json"

	if err := model.Save(path); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadSTTModel(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.CTC == nil {
		t.Fatal("loaded CTC is nil")
	}

	if loaded.CTC.Vocabulary == nil {
		t.Fatal("loaded vocabulary is nil")
	}

	tokens := loaded.CTC.Vocabulary.Tokens()

	if len(tokens) != len(vocabulary.Tokens()) {
		t.Fatalf(
			"vocabulary size mismatch: got=%d want=%d",
			len(tokens),
			len(vocabulary.Tokens()),
		)
	}

	for i, token := range vocabulary.Tokens() {
		if tokens[i] != token {
			t.Fatalf(
				"vocabulary token mismatch at %d: got=%q want=%q",
				i,
				tokens[i],
				token,
			)
		}
	}

	if loaded.CTC.BlankID != vocabulary.BlankID() {
		t.Fatalf(
			"blank id mismatch: got=%d want=%d",
			loaded.CTC.BlankID,
			vocabulary.BlankID(),
		)
	}
	after, err := loaded.Transcribe(
		input,
		inputLengths,
	)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf(
		"After load: %q",
		after,
	)

	if len(before) != len(after) {
		t.Fatalf(
			"transcription count changed: before=%v after=%v",
			before,
			after,
		)
	}

	for i := range before {
		if before[i] != after[i] {
			t.Fatalf(
				"transcription changed: before=%v after=%v",
				before,
				after,
			)
		}
	}
}
func TestDataLoader(t *testing.T) {
	dataset := NewDataset()

	for i := 0; i < 5; i++ {
		rows := 2 + i

		features := NewFeatureMatrix(rows, 3)

		for row := 0; row < rows; row++ {
			for col := 0; col < 3; col++ {
				features.Set(
					row,
					col,
					float32(i*100+row*10+col),
				)
			}
		}

		err := dataset.Add(TrainingSample{
			Features: features,
			Labels:   []int{i + 1},
			Text:     "sample",
		})

		if err != nil {
			t.Fatal(err)
		}
	}

	loader, err := NewDataLoader(
		dataset,
		2,
		false,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatal(err)
	}

	batchCount := 0
	totalSamples := 0

	for {
		batch, ok, err := loader.Next()
		if err != nil {
			t.Fatal(err)
		}

		if !ok {
			break
		}

		batchCount++
		totalSamples += batch.BatchSize

		t.Logf(
			"batch=%d size=%d time=%d features=%d",
			batchCount,
			batch.BatchSize,
			batch.TimeSteps,
			batch.FeatureDim,
		)
	}

	if batchCount != 3 {
		t.Fatalf(
			"batch count mismatch: got=%d want=3",
			batchCount,
		)
	}

	if totalSamples != 5 {
		t.Fatalf(
			"sample count mismatch: got=%d want=5",
			totalSamples,
		)
	}
}
func TestTrainerTrain(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}
	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)
	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}
	dataset := NewDataset()
	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}
	labels, err := tokenizer.Encode("hel")
	if err != nil {
		t.Fatal(err)
	}
	for sampleIndex := 0; sampleIndex < 4; sampleIndex++ {
		features := NewFeatureMatrix(
			20,
			80,
		)
		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				value := float32(
					(row+1)*(col+1),
				) * 0.0001
				value += float32(sampleIndex) * 0.00001
				features.Set(
					row,
					col,
					value,
				)
			}
		}
		err := dataset.Add(
			TrainingSample{
				Features: features,
				Labels:   append([]int(nil), labels...),
				Text:     "hel",
			},
		)
		if err != nil {
			t.Fatal(err)
		}
	}
	config := TrainingConfig{
		Epochs:    10,
		BatchSize: 2,
	}
	losses, err := trainer.Train(
		dataset,
		config,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(losses) != config.Epochs {
		t.Fatalf(
			"loss count mismatch: got=%d want=%d",
			len(losses),
			config.Epochs,
		)
	}
	for epoch, loss := range losses {
		if math.IsNaN(float64(loss)) ||
			math.IsInf(float64(loss), 0) {
			t.Fatalf(
				"invalid loss at epoch %d: %f",
				epoch,
				loss,
			)
		}
		if loss <= 0 {
			t.Fatalf(
				"invalid loss at epoch %d: %f",
				epoch,
				loss,
			)
		}
		t.Logf(
			"epoch=%d loss=%.6f",
			epoch,
			loss,
		)
	}
	if losses[len(losses)-1] >= losses[0] {
		t.Fatalf(
			"loss did not decrease: first=%f last=%f",
			losses[0],
			losses[len(losses)-1],
		)
	}
}
func TestTrainerTrainValidation(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		4,
		vocabulary,
	)

	optimizer, err := NewAdam(0.01)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	features := NewFeatureMatrix(10, 80)

	err = dataset.Add(TrainingSample{
		Features: features,
		Labels:   []int{8, 5, 12},
		Text:     "hel",
	})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name   string
		data   *Dataset
		config TrainingConfig
	}{
		{
			name: "nil dataset",
			data: nil,
			config: TrainingConfig{
				Epochs:    1,
				BatchSize: 1,
			},
		},
		{
			name: "empty dataset",
			data: NewDataset(),
			config: TrainingConfig{
				Epochs:    1,
				BatchSize: 1,
			},
		},
		{
			name: "zero epochs",
			data: dataset,
			config: TrainingConfig{
				Epochs:    0,
				BatchSize: 1,
			},
		},
		{
			name: "negative epochs",
			data: dataset,
			config: TrainingConfig{
				Epochs:    -1,
				BatchSize: 1,
			},
		},
		{
			name: "zero batch size",
			data: dataset,
			config: TrainingConfig{
				Epochs:    1,
				BatchSize: 0,
			},
		},
		{
			name: "negative batch size",
			data: dataset,
			config: TrainingConfig{
				Epochs:    1,
				BatchSize: -1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			losses, err := trainer.Train(
				tt.data,
				tt.config,
			)

			if err == nil {
				t.Fatalf(
					"expected error, got nil with losses=%v",
					losses,
				)
			}
		})
	}
}
func TestTrainerTrainNilReceiver(t *testing.T) {
	var trainer *Trainer

	dataset := NewDataset()

	losses, err := trainer.Train(
		dataset,
		TrainingConfig{
			Epochs:    1,
			BatchSize: 1,
		},
	)

	if err == nil {
		t.Fatalf(
			"expected error, got nil with losses=%v",
			losses,
		)
	}
}
func TestSTTModelEvaluateBatch(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	if model == nil {
		t.Fatal("model is nil")
	}

	dataset := NewDataset()

	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}

	labels, err := tokenizer.Encode("hel")
	if err != nil {
		t.Fatal(err)
	}

	features := NewFeatureMatrix(20, 80)

	for row := 0; row < 20; row++ {
		for col := 0; col < 80; col++ {
			value := float32(
				(row+1)*(col+1),
			) * 0.0001

			features.Set(row, col, value)
		}
	}

	err = dataset.Add(
		TrainingSample{
			Features: features,
			Labels:   labels,
			Text:     "hel",
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	batch, err := CreateBatch(
		dataset,
		0,
		1,
	)
	if err != nil {
		t.Fatal(err)
	}

	input, err := batch.ToTensor()
	if err != nil {
		t.Fatal(err)
	}

	loss, err := model.EvaluateBatch(
		input,
		batch.InputLengths,
		batch.Labels,
		batch.LabelLengths,
	)
	if err != nil {
		t.Fatal(err)
	}

	if math.IsNaN(float64(loss)) ||
		math.IsInf(float64(loss), 0) {
		t.Fatalf(
			"invalid evaluation loss: %f",
			loss,
		)
	}

	if loss <= 0 {
		t.Fatalf(
			"invalid evaluation loss: %f",
			loss,
		)
	}

	t.Logf(
		"Evaluation loss: %.6f",
		loss,
	)
}
func TestTrainerEvaluate(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}

	labels, err := tokenizer.Encode("hel")
	if err != nil {
		t.Fatal(err)
	}

	for sampleIndex := 0; sampleIndex < 4; sampleIndex++ {
		features := NewFeatureMatrix(20, 80)

		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				value := float32(
					(row+1)*(col+1),
				) * 0.0001

				value += float32(sampleIndex) * 0.00001

				features.Set(
					row,
					col,
					value,
				)
			}
		}

		err := dataset.Add(TrainingSample{
			Features: features,
			Labels:   append([]int(nil), labels...),
			Text:     "hel",
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	loader, err := NewDataLoader(
		dataset,
		2,
		false,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatal(err)
	}

	loss, err := trainer.Evaluate(loader)
	if err != nil {
		t.Fatal(err)
	}

	if math.IsNaN(float64(loss)) ||
		math.IsInf(float64(loss), 0) {
		t.Fatalf(
			"invalid evaluation loss: %f",
			loss,
		)
	}

	if loss <= 0 {
		t.Fatalf(
			"invalid evaluation loss: %f",
			loss,
		)
	}

	t.Logf(
		"Evaluation loss: %.6f",
		loss,
	)
}
func TestTrainerEvaluateDoesNotUpdateModel(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}

	labels, err := tokenizer.Encode("hel")
	if err != nil {
		t.Fatal(err)
	}

	features := NewFeatureMatrix(20, 80)

	for row := 0; row < 20; row++ {
		for col := 0; col < 80; col++ {
			features.Set(
				row,
				col,
				float32((row+1)*(col+1))*0.0001,
			)
		}
	}

	err = dataset.Add(TrainingSample{
		Features: features,
		Labels:   labels,
		Text:     "hel",
	})
	if err != nil {
		t.Fatal(err)
	}

	loader, err := NewDataLoader(
		dataset,
		1,
		false,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatal(err)
	}

	beforeDecoderWeights := append(
		[]float32(nil),
		model.Decoder.Weights...,
	)

	beforeDecoderBias := append(
		[]float32(nil),
		model.Decoder.Bias...,
	)

	beforeForwardWeights := append(
		[]float32(nil),
		model.Encoder.ForwardLSTM.Cell.Weights...,
	)

	beforeForwardBias := append(
		[]float32(nil),
		model.Encoder.ForwardLSTM.Cell.Bias...,
	)

	beforeBackwardWeights := append(
		[]float32(nil),
		model.Encoder.BackwardLSTM.Cell.Weights...,
	)

	beforeBackwardBias := append(
		[]float32(nil),
		model.Encoder.BackwardLSTM.Cell.Bias...,
	)

	beforeStepCount := optimizer.StepCount

	loss, err := trainer.Evaluate(loader)
	if err != nil {
		t.Fatal(err)
	}

	if math.IsNaN(float64(loss)) ||
		math.IsInf(float64(loss), 0) {
		t.Fatalf(
			"invalid evaluation loss: %f",
			loss,
		)
	}

	if !reflect.DeepEqual(
		model.Decoder.Weights,
		beforeDecoderWeights,
	) {
		t.Fatal("decoder weights changed during evaluation")
	}

	if !reflect.DeepEqual(
		model.Decoder.Bias,
		beforeDecoderBias,
	) {
		t.Fatal("decoder bias changed during evaluation")
	}

	if !reflect.DeepEqual(
		model.Encoder.ForwardLSTM.Cell.Weights,
		beforeForwardWeights,
	) {
		t.Fatal("forward LSTM weights changed during evaluation")
	}

	if !reflect.DeepEqual(
		model.Encoder.ForwardLSTM.Cell.Bias,
		beforeForwardBias,
	) {
		t.Fatal("forward LSTM bias changed during evaluation")
	}

	if !reflect.DeepEqual(
		model.Encoder.BackwardLSTM.Cell.Weights,
		beforeBackwardWeights,
	) {
		t.Fatal("backward LSTM weights changed during evaluation")
	}

	if !reflect.DeepEqual(
		model.Encoder.BackwardLSTM.Cell.Bias,
		beforeBackwardBias,
	) {
		t.Fatal("backward LSTM bias changed during evaluation")
	}

	if optimizer.StepCount != beforeStepCount {
		t.Fatalf(
			"optimizer step count changed: before=%d after=%d",
			beforeStepCount,
			optimizer.StepCount,
		)
	}

	t.Logf(
		"Evaluation loss: %.6f",
		loss,
	)
}
func TestCharacterErrorRate(t *testing.T) {

	tests := []struct {
		name       string
		reference  string
		hypothesis string
		expected   float32
	}{
		{
			name:       "identical",
			reference:  "hello",
			hypothesis: "hello",
			expected:   0,
		},
		{
			name:       "deletion",
			reference:  "hello",
			hypothesis: "helo",
			expected:   0.2,
		},
		{
			name:       "insertion",
			reference:  "helo",
			hypothesis: "hello",
			expected:   0.25,
		},
		{
			name:       "substitution",
			reference:  "hello",
			hypothesis: "hallo",
			expected:   0.2,
		},
		{
			name:       "empty_hypothesis",
			reference:  "hello",
			hypothesis: "",
			expected:   1,
		},
		{
			name:       "unicode",
			reference:  "سلام",
			hypothesis: "سلم",
			expected:   0.25,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := CharacterErrorRate(
				test.reference,
				test.hypothesis,
			)

			if err != nil {
				t.Fatal(err)
			}

			if math.Abs(float64(got-test.expected)) > 1e-6 {
				t.Fatalf(
					"expected %.4f, got %.4f",
					test.expected,
					got,
				)
			}
		})
	}
}
func TestCharacterErrorRateEmptyReference(t *testing.T) {

	_, err := CharacterErrorRate("", "hello")

	if !errors.Is(err, ErrInvalidMetricInput) {
		t.Fatalf(
			"expected ErrInvalidMetricInput, got %v",
			err,
		)
	}
}
func TestSTTModelEvaluateBatchResult(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	if model == nil {
		t.Fatal("expected model")
	}

	input := NewTensor3D(1, 6, 80)

	inputLengths := []int{6}
	targets := []int{8, 5, 12}
	targetLengths := []int{3}

	result, err := model.EvaluateBatchResult(
		input,
		inputLengths,
		targets,
		targetLengths,
	)

	if err != nil {
		t.Fatal(err)
	}

	if result == nil {
		t.Fatal("expected result")
	}

	if math.IsNaN(float64(result.Loss)) ||
		math.IsInf(float64(result.Loss), 0) {
		t.Fatalf("invalid loss: %v", result.Loss)
	}

	if len(result.Predictions) != 1 {
		t.Fatalf(
			"expected 1 prediction, got %d",
			len(result.Predictions),
		)
	}

	t.Logf(
		"Loss: %f",
		result.Loss,
	)

	t.Logf(
		"Predictions: %q",
		result.Predictions,
	)
}
func TestTrainerEvaluateWithMetrics(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	labels := []int{8, 5, 12}

	for i := 0; i < 4; i++ {
		features := NewFeatureMatrix(20, 80)

		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				features.Set(
					row,
					col,
					float32((row+1)*(col+1))*0.0001,
				)
			}
		}

		err := dataset.Add(TrainingSample{
			Features: features,
			Labels:   append([]int(nil), labels...),
			Text:     "hel",
		})

		if err != nil {
			t.Fatal(err)
		}
	}

	loader, err := NewDataLoader(
		dataset,
		2,
		false,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := trainer.EvaluateWithMetrics(loader)
	if err != nil {
		t.Fatal(err)
	}

	if result == nil {
		t.Fatal("expected evaluation result")
	}

	if math.IsNaN(float64(result.Loss)) ||
		math.IsInf(float64(result.Loss), 0) {
		t.Fatalf("invalid loss: %v", result.Loss)
	}

	if math.IsNaN(float64(result.CER)) ||
		math.IsInf(float64(result.CER), 0) {
		t.Fatalf("invalid CER: %v", result.CER)
	}

	if result.Loss <= 0 {
		t.Fatalf("expected positive loss, got %v", result.Loss)
	}

	if result.CER < 0 {
		t.Fatalf("invalid CER: %v", result.CER)
	}
	if result.WER < 0 {
		t.Fatalf(
			"invalid WER: %v",
			result.WER,
		)
	}
	t.Logf(
		"Evaluation loss: %f",
		result.Loss,
	)

	t.Logf(
		"Evaluation CER: %f",
		result.CER,
	)
	t.Logf(
		"Evaluation WER: %f",
		result.WER,
	)
}
func TestTrainerEvaluateWithMetricsDoesNotUpdateModel(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(80, 8, vocabulary)

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(model, optimizer)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	features := NewFeatureMatrix(20, 80)

	for row := 0; row < 20; row++ {
		for col := 0; col < 80; col++ {
			features.Set(
				row,
				col,
				float32((row+1)*(col+1))*0.0001,
			)
		}
	}

	if err := dataset.Add(TrainingSample{
		Features: features,
		Labels:   []int{8, 5, 12},
		Text:     "hel",
	}); err != nil {
		t.Fatal(err)
	}

	loader, err := NewDataLoader(
		dataset,
		1,
		false,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	decoderWeights := append(
		[]float32(nil),
		model.Decoder.Weights...,
	)

	decoderBias := append(
		[]float32(nil),
		model.Decoder.Bias...,
	)

	forwardWeights := append(
		[]float32(nil),
		model.Encoder.ForwardLSTM.Cell.Weights...,
	)

	forwardBias := append(
		[]float32(nil),
		model.Encoder.ForwardLSTM.Cell.Bias...,
	)

	backwardWeights := append(
		[]float32(nil),
		model.Encoder.BackwardLSTM.Cell.Weights...,
	)

	backwardBias := append(
		[]float32(nil),
		model.Encoder.BackwardLSTM.Cell.Bias...,
	)

	stepCount := optimizer.StepCount

	_, err = trainer.EvaluateWithMetrics(loader)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(
		decoderWeights,
		model.Decoder.Weights,
	) {
		t.Fatal("decoder weights changed")
	}

	if !reflect.DeepEqual(
		decoderBias,
		model.Decoder.Bias,
	) {
		t.Fatal("decoder bias changed")
	}

	if !reflect.DeepEqual(
		forwardWeights,
		model.Encoder.ForwardLSTM.Cell.Weights,
	) {
		t.Fatal("forward LSTM weights changed")
	}

	if !reflect.DeepEqual(
		forwardBias,
		model.Encoder.ForwardLSTM.Cell.Bias,
	) {
		t.Fatal("forward LSTM bias changed")
	}

	if !reflect.DeepEqual(
		backwardWeights,
		model.Encoder.BackwardLSTM.Cell.Weights,
	) {
		t.Fatal("backward LSTM weights changed")
	}

	if !reflect.DeepEqual(
		backwardBias,
		model.Encoder.BackwardLSTM.Cell.Bias,
	) {
		t.Fatal("backward LSTM bias changed")
	}

	if optimizer.StepCount != stepCount {
		t.Fatalf(
			"optimizer step count changed: before=%d after=%d",
			stepCount,
			optimizer.StepCount,
		)
	}
}
func TestWordErrorRate(t *testing.T) {

	tests := []struct {
		name       string
		reference  string
		hypothesis string
		expected   float32
	}{
		{
			name:       "identical",
			reference:  "hello world",
			hypothesis: "hello world",
			expected:   0,
		},
		{
			name:       "deletion",
			reference:  "hello world",
			hypothesis: "hello",
			expected:   0.5,
		},
		{
			name:       "insertion",
			reference:  "hello",
			hypothesis: "hello world",
			expected:   1,
		},
		{
			name:       "substitution",
			reference:  "hello world",
			hypothesis: "hello there",
			expected:   0.5,
		},
		{
			name:       "empty_hypothesis",
			reference:  "hello world",
			hypothesis: "",
			expected:   1,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			got, err := WordErrorRate(
				test.reference,
				test.hypothesis,
			)

			if err != nil {
				t.Fatal(err)
			}

			if math.Abs(
				float64(got-test.expected),
			) > 1e-6 {
				t.Fatalf(
					"expected %.4f, got %.4f",
					test.expected,
					got,
				)
			}
		})
	}
}
func TestWordErrorRateEmptyReference(t *testing.T) {

	_, err := WordErrorRate(
		"",
		"hello",
	)

	if !errors.Is(err, ErrInvalidMetricInput) {
		t.Fatalf(
			"expected ErrInvalidMetricInput, got %v",
			err,
		)
	}
}
func TestSplitDataset(t *testing.T) {

	dataset := NewDataset()

	for i := 0; i < 10; i++ {

		features := NewFeatureMatrix(5, 80)

		err := dataset.Add(TrainingSample{
			Features: features,
			Labels:   []int{8, 5, 12},
			Text:     "hel",
		})

		if err != nil {
			t.Fatal(err)
		}
	}

	split, err := SplitDataset(
		dataset,
		0.8,
		0,
		0.2,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatal(err)
	}

	if split.Train.Len() != 8 {
		t.Fatalf(
			"expected 8 train samples, got %d",
			split.Train.Len(),
		)
	}

	if split.Test.Len() != 2 {
		t.Fatalf(
			"expected 2 test samples, got %d",
			split.Test.Len(),
		)
	}

	if split.Validation.Len() != 0 {
		t.Fatalf(
			"expected 0 validation samples, got %d",
			split.Validation.Len(),
		)
	}

	t.Logf(
		"Train=%d Validation=%d Test=%d",
		split.Train.Len(),
		split.Validation.Len(),
		split.Test.Len(),
	)
}
func TestSplitDatasetDeterministic(t *testing.T) {

	dataset := NewDataset()

	for i := 0; i < 10; i++ {

		features := NewFeatureMatrix(5, 80)

		err := dataset.Add(TrainingSample{
			Features: features,
			Labels:   []int{8, 5, 12},
			Text:     fmt.Sprintf("sample-%d", i),
		})

		if err != nil {
			t.Fatal(err)
		}
	}

	split1, err := SplitDataset(
		dataset,
		0.6,
		0.2,
		0.2,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatal(err)
	}

	split2, err := SplitDataset(
		dataset,
		0.6,
		0.2,
		0.2,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatal(err)
	}

	if split1.Train.Len() != 6 ||
		split1.Validation.Len() != 2 ||
		split1.Test.Len() != 2 {
		t.Fatalf(
			"unexpected split sizes: train=%d validation=%d test=%d",
			split1.Train.Len(),
			split1.Validation.Len(),
			split1.Test.Len(),
		)
	}

	getTexts := func(dataset *Dataset) []string {

		texts := make([]string, dataset.Len())

		for i := 0; i < dataset.Len(); i++ {
			sample, err := dataset.Get(i)
			if err != nil {
				t.Fatal(err)
			}

			texts[i] = sample.Text
		}

		return texts
	}

	train1 := getTexts(split1.Train)
	train2 := getTexts(split2.Train)

	validation1 := getTexts(split1.Validation)
	validation2 := getTexts(split2.Validation)

	test1 := getTexts(split1.Test)
	test2 := getTexts(split2.Test)

	if !reflect.DeepEqual(train1, train2) {
		t.Fatal("train split is not deterministic")
	}

	if !reflect.DeepEqual(validation1, validation2) {
		t.Fatal("validation split is not deterministic")
	}

	if !reflect.DeepEqual(test1, test2) {
		t.Fatal("test split is not deterministic")
	}

	seen := make(map[string]string)

	checkSplit := func(
		name string,
		texts []string,
	) {
		for _, text := range texts {
			if previous, exists := seen[text]; exists {
				t.Fatalf(
					"sample %q appears in both %s and %s",
					text,
					previous,
					name,
				)
			}

			seen[text] = name
		}
	}

	checkSplit("train", train1)
	checkSplit("validation", validation1)
	checkSplit("test", test1)

	if len(seen) != dataset.Len() {
		t.Fatalf(
			"expected %d unique samples, got %d",
			dataset.Len(),
			len(seen),
		)
	}
}
func TestTrainerTrainWithValidation(t *testing.T) {

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	for i := 0; i < 10; i++ {

		features := NewFeatureMatrix(20, 80)

		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				features.Set(
					row,
					col,
					float32((row+1)*(col+1))*0.0001,
				)
			}
		}

		if err := dataset.Add(TrainingSample{
			Features: features,
			Labels:   []int{8, 5, 12},
			Text:     "hel",
		}); err != nil {
			t.Fatal(err)
		}
	}

	split, err := SplitDataset(
		dataset,
		0.8,
		0,
		0.2,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatal(err)
	}

	history, err := trainer.TrainWithValidation(
		split.Train,
		split.Test,
		TrainingConfig{
			Epochs:    5,
			BatchSize: 2,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	if history == nil {
		t.Fatal("expected training history")
	}

	if len(history.TrainLoss) != 5 {
		t.Fatalf(
			"expected 5 train losses, got %d",
			len(history.TrainLoss),
		)
	}

	if len(history.ValidationLoss) != 5 {
		t.Fatalf(
			"expected 5 validation losses, got %d",
			len(history.ValidationLoss),
		)
	}

	if len(history.ValidationCER) != 5 {
		t.Fatalf(
			"expected 5 validation CER values, got %d",
			len(history.ValidationCER),
		)
	}

	if len(history.ValidationWER) != 5 {
		t.Fatalf(
			"expected 5 validation WER values, got %d",
			len(history.ValidationWER),
		)
	}

	for epoch := 0; epoch < 5; epoch++ {

		if math.IsNaN(
			float64(history.TrainLoss[epoch]),
		) || math.IsInf(
			float64(history.TrainLoss[epoch]),
			0,
		) {
			t.Fatalf(
				"invalid train loss at epoch %d: %v",
				epoch,
				history.TrainLoss[epoch],
			)
		}

		if math.IsNaN(
			float64(history.ValidationLoss[epoch]),
		) || math.IsInf(
			float64(history.ValidationLoss[epoch]),
			0,
		) {
			t.Fatalf(
				"invalid validation loss at epoch %d: %v",
				epoch,
				history.ValidationLoss[epoch],
			)
		}
	}

	t.Logf(
		"epoch=0 train=%.6f validation=%.6f CER=%.6f WER=%.6f",
		history.TrainLoss[0],
		history.ValidationLoss[0],
		history.ValidationCER[0],
		history.ValidationWER[0],
	)

	t.Logf(
		"epoch=4 train=%.6f validation=%.6f CER=%.6f WER=%.6f",
		history.TrainLoss[4],
		history.ValidationLoss[4],
		history.ValidationCER[4],
		history.ValidationWER[4],
	)
}
func TestEarlyStopping(t *testing.T) {

	stopping, err := NewEarlyStopping(
		2,
		0.01,
	)
	if err != nil {
		t.Fatal(err)
	}

	stop, err := stopping.Update(10)
	if err != nil {
		t.Fatal(err)
	}

	if stop {
		t.Fatal("should not stop on first loss")
	}

	stop, err = stopping.Update(9)
	if err != nil {
		t.Fatal(err)
	}

	if stop {
		t.Fatal("should not stop after improvement")
	}

	stop, err = stopping.Update(9.5)
	if err != nil {
		t.Fatal(err)
	}

	if stop {
		t.Fatal("should not stop before patience")
	}

	stop, err = stopping.Update(9.6)
	if err != nil {
		t.Fatal(err)
	}

	if !stop {
		t.Fatal("expected early stopping")
	}

	if stopping.BestLoss() != 9 {
		t.Fatalf(
			"expected best loss 9, got %f",
			stopping.BestLoss(),
		)
	}

	if stopping.WaitCount() != 2 {
		t.Fatalf(
			"expected wait count 2, got %d",
			stopping.WaitCount(),
		)
	}
}
func TestEarlyStoppingValidation(t *testing.T) {

	_, err := NewEarlyStopping(0, 0)

	if !errors.Is(err, ErrInvalidEarlyStopping) {
		t.Fatalf(
			"expected ErrInvalidEarlyStopping, got %v",
			err,
		)
	}

	_, err = NewEarlyStopping(2, -1)

	if !errors.Is(err, ErrInvalidEarlyStopping) {
		t.Fatalf(
			"expected ErrInvalidEarlyStopping, got %v",
			err,
		)
	}
}
func TestSTTModelSnapshotRestore(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	if model == nil {
		t.Fatal("expected model")
	}

	// Give the model deterministic non-zero values.
	for i := range model.Decoder.Weights {
		model.Decoder.Weights[i] = float32(i+1) * 0.001
	}

	for i := range model.Decoder.Bias {
		model.Decoder.Bias[i] = float32(i+1) * 0.002
	}

	for i := range model.Encoder.ForwardLSTM.Cell.Weights {
		model.Encoder.ForwardLSTM.Cell.Weights[i] =
			float32(i+1) * 0.003
	}

	for i := range model.Encoder.ForwardLSTM.Cell.Bias {
		model.Encoder.ForwardLSTM.Cell.Bias[i] =
			float32(i+1) * 0.004
	}

	for i := range model.Encoder.BackwardLSTM.Cell.Weights {
		model.Encoder.BackwardLSTM.Cell.Weights[i] =
			float32(i+1) * 0.005
	}

	for i := range model.Encoder.BackwardLSTM.Cell.Bias {
		model.Encoder.BackwardLSTM.Cell.Bias[i] =
			float32(i+1) * 0.006
	}

	snapshot, err := model.Snapshot()
	if err != nil {
		t.Fatal(err)
	}

	// Change the model after taking the snapshot.
	for i := range model.Decoder.Weights {
		model.Decoder.Weights[i] = 999
	}

	for i := range model.Decoder.Bias {
		model.Decoder.Bias[i] = 999
	}

	for i := range model.Encoder.ForwardLSTM.Cell.Weights {
		model.Encoder.ForwardLSTM.Cell.Weights[i] = 999
	}

	for i := range model.Encoder.ForwardLSTM.Cell.Bias {
		model.Encoder.ForwardLSTM.Cell.Bias[i] = 999
	}

	for i := range model.Encoder.BackwardLSTM.Cell.Weights {
		model.Encoder.BackwardLSTM.Cell.Weights[i] = 999
	}

	for i := range model.Encoder.BackwardLSTM.Cell.Bias {
		model.Encoder.BackwardLSTM.Cell.Bias[i] = 999
	}

	err = model.Restore(snapshot)
	if err != nil {
		t.Fatal(err)
	}

	// Verify decoder.
	for i, value := range model.Decoder.Weights {
		expected := float32(i+1) * 0.001

		if value != expected {
			t.Fatalf(
				"decoder weight[%d]: expected %f, got %f",
				i,
				expected,
				value,
			)
		}
	}

	for i, value := range model.Decoder.Bias {
		expected := float32(i+1) * 0.002

		if value != expected {
			t.Fatalf(
				"decoder bias[%d]: expected %f, got %f",
				i,
				expected,
				value,
			)
		}
	}

	// Verify forward LSTM.
	for i, value := range model.Encoder.ForwardLSTM.Cell.Weights {
		expected := float32(i+1) * 0.003

		if value != expected {
			t.Fatalf(
				"forward weight[%d]: expected %f, got %f",
				i,
				expected,
				value,
			)
		}
	}

	for i, value := range model.Encoder.ForwardLSTM.Cell.Bias {
		expected := float32(i+1) * 0.004

		if value != expected {
			t.Fatalf(
				"forward bias[%d]: expected %f, got %f",
				i,
				expected,
				value,
			)
		}
	}

	// Verify backward LSTM.
	for i, value := range model.Encoder.BackwardLSTM.Cell.Weights {
		expected := float32(i+1) * 0.005

		if value != expected {
			t.Fatalf(
				"backward weight[%d]: expected %f, got %f",
				i,
				expected,
				value,
			)
		}
	}

	for i, value := range model.Encoder.BackwardLSTM.Cell.Bias {
		expected := float32(i+1) * 0.006

		if value != expected {
			t.Fatalf(
				"backward bias[%d]: expected %f, got %f",
				i,
				expected,
				value,
			)
		}
	}
}
func TestSTTModelSnapshotIsIndependent(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	if model == nil {
		t.Fatal("expected model")
	}

	model.Decoder.Weights[0] = 1
	model.Decoder.Bias[0] = 2

	snapshot, err := model.Snapshot()
	if err != nil {
		t.Fatal(err)
	}

	model.Decoder.Weights[0] = 100
	model.Decoder.Bias[0] = 200

	if snapshot.DecoderWeights[0] != 1 {
		t.Fatalf(
			"snapshot weight changed: got %f",
			snapshot.DecoderWeights[0],
		)
	}

	if snapshot.DecoderBias[0] != 2 {
		t.Fatalf(
			"snapshot bias changed: got %f",
			snapshot.DecoderBias[0],
		)
	}
}
func TestTrainerTrainWithValidationBestModel(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()
	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		features := NewFeatureMatrix(20, 80)

		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				features.Set(
					row,
					col,
					float32((row+1)*(col+1))*0.0001,
				)
			}
		}

		labels, err := tokenizer.Encode("hel")
		if err != nil {
			t.Fatal(err)
		}

		err = dataset.Add(
			TrainingSample{
				Features: features,
				Labels:   labels,
				Text:     "hel",
			},
		)
		if err != nil {
			t.Fatal(err)
		}
	}

	split, err := SplitDataset(
		dataset,
		0.8,
		0,
		0.2,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatal(err)
	}

	config := TrainingConfig{
		Epochs:    10,
		BatchSize: 2,
	}

	history, err := trainer.TrainWithValidation(
		split.Train,
		split.Test,
		config,
	)
	if err != nil {
		t.Fatal(err)
	}

	if history.BestEpoch < 0 {
		t.Fatal("expected valid best epoch")
	}

	if history.BestEpoch >= len(history.ValidationLoss) {
		t.Fatalf(
			"invalid best epoch: %d",
			history.BestEpoch,
		)
	}

	if history.BestLoss != history.ValidationLoss[history.BestEpoch] {
		t.Fatalf(
			"best loss mismatch: best=%f validation=%f",
			history.BestLoss,
			history.ValidationLoss[history.BestEpoch],
		)
	}

	for i, loss := range history.ValidationLoss {
		if loss < history.BestLoss {
			t.Fatalf(
				"epoch %d has better loss %f than best %f",
				i,
				loss,
				history.BestLoss,
			)
		}
	}
}
func TestTrainingHistoryBestLoss(t *testing.T) {
	history := &TrainingHistory{
		ValidationLoss: []float32{
			10,
			7,
			5,
			6,
			8,
		},
		BestEpoch: 2,
		BestLoss:  5,
	}

	if history.BestEpoch != 2 {
		t.Fatalf(
			"expected best epoch 2, got %d",
			history.BestEpoch,
		)
	}

	if history.BestLoss != history.ValidationLoss[history.BestEpoch] {
		t.Fatalf(
			"best loss mismatch",
		)
	}

	for i, loss := range history.ValidationLoss {
		if loss < history.BestLoss {
			t.Fatalf(
				"epoch %d has lower loss %f than best %f",
				i,
				loss,
				history.BestLoss,
			)
		}
	}
}
func TestTrainerTrainWithValidationRestoresBestModel(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	for i := 0; i < 10; i++ {
		features := NewFeatureMatrix(20, 80)

		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				features.Set(
					row,
					col,
					float32((row+1)*(col+1))*0.0001,
				)
			}
		}

		labels, err := tokenizer.Encode("hel")
		if err != nil {
			t.Fatal(err)
		}

		err = dataset.Add(
			TrainingSample{
				Features: features,
				Labels:   labels,
				Text:     "hel",
			},
		)
		if err != nil {
			t.Fatal(err)
		}
	}

	split, err := SplitDataset(
		dataset,
		0.8,
		0,
		0.2,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	config := TrainingConfig{
		Epochs:    10,
		BatchSize: 2,
	}

	history, err := trainer.TrainWithValidation(
		split.Train,
		split.Test,
		config,
	)
	if err != nil {
		t.Fatal(err)
	}

	if history.BestEpoch < 0 {
		t.Fatal("expected valid best epoch")
	}

	validationLoader, err := NewDataLoader(
		split.Test,
		config.BatchSize,
		false,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	finalLoss, err := trainer.Evaluate(
		validationLoader,
	)
	if err != nil {
		t.Fatal(err)
	}

	const tolerance float32 = 1e-5

	difference := finalLoss - history.BestLoss

	if difference < 0 {
		difference = -difference
	}

	if difference > tolerance {
		t.Fatalf(
			"model was not restored to best state: best=%f final=%f diff=%f",
			history.BestLoss,
			finalLoss,
			difference,
		)
	}

	t.Logf(
		"Best epoch: %d",
		history.BestEpoch,
	)

	t.Logf(
		"Best loss: %.6f",
		history.BestLoss,
	)

	t.Logf(
		"Final restored loss: %.6f",
		finalLoss,
	)
}
func TestAdamSnapshotRestore(t *testing.T) {
	adam, err := NewAdam(0.01)
	if err != nil {
		t.Fatal(err)
	}

	parameters := []*Parameter{
		{
			Name:   "weight",
			Values: []float32{1, 2, 3},
			Gradients: []float32{
				0.1,
				0.2,
				0.3,
			},
		},
	}

	for i := 0; i < 5; i++ {
		if err := adam.Step(parameters); err != nil {
			t.Fatal(err)
		}
	}

	checkpoint, err := adam.Snapshot()
	if err != nil {
		t.Fatal(err)
	}

	if checkpoint.StepCount != 5 {
		t.Fatalf(
			"expected step count 5, got %d",
			checkpoint.StepCount,
		)
	}

	state, ok := checkpoint.States["weight"]
	if !ok {
		t.Fatal("expected weight optimizer state")
	}

	if len(state.FirstMoment) != 3 {
		t.Fatalf(
			"expected first moment length 3, got %d",
			len(state.FirstMoment),
		)
	}

	if len(state.SecondMoment) != 3 {
		t.Fatalf(
			"expected second moment length 3, got %d",
			len(state.SecondMoment),
		)
	}

	// Change optimizer.
	adam.StepCount = 999
	adam.States["weight"].FirstMoment[0] = 999

	err = adam.Restore(checkpoint)
	if err != nil {
		t.Fatal(err)
	}

	if adam.StepCount != 5 {
		t.Fatalf(
			"expected restored step count 5, got %d",
			adam.StepCount,
		)
	}

	if adam.States["weight"].FirstMoment[0] !=
		checkpoint.States["weight"].FirstMoment[0] {
		t.Fatal("first moment was not restored")
	}
}
func TestAdamSnapshotIsIndependent(t *testing.T) {
	adam, err := NewAdam(0.01)
	if err != nil {
		t.Fatal(err)
	}

	parameters := []*Parameter{
		{
			Name:   "weight",
			Values: []float32{1, 2},
			Gradients: []float32{
				0.1,
				0.2,
			},
		},
	}

	if err := adam.Step(parameters); err != nil {
		t.Fatal(err)
	}

	checkpoint, err := adam.Snapshot()
	if err != nil {
		t.Fatal(err)
	}

	adam.States["weight"].FirstMoment[0] = 999

	if checkpoint.States["weight"].FirstMoment[0] == 999 {
		t.Fatal("checkpoint shares first moment memory")
	}
}
func TestTrainerCheckpointRestore(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	optimizer, err := NewAdam(0.01)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	parameters := []*Parameter{
		{
			Name:   "test",
			Values: []float32{1, 2},
			Gradients: []float32{
				0.1,
				0.2,
			},
		},
	}

	if err := optimizer.Step(parameters); err != nil {
		t.Fatal(err)
	}

	model.Decoder.Weights[0] = 123

	checkpoint, err := trainer.CreateCheckpoint(
		5,
		10,
	)
	if err != nil {
		t.Fatal(err)
	}

	if checkpoint.Epoch != 5 {
		t.Fatalf(
			"expected epoch 5, got %d",
			checkpoint.Epoch,
		)
	}

	if checkpoint.Step != 10 {
		t.Fatalf(
			"expected step 10, got %d",
			checkpoint.Step,
		)
	}

	// Destroy current state.
	model.Decoder.Weights[0] = 999
	optimizer.StepCount = 999

	err = trainer.RestoreCheckpoint(checkpoint)
	if err != nil {
		t.Fatal(err)
	}

	if model.Decoder.Weights[0] != 123 {
		t.Fatalf(
			"model was not restored: got %f",
			model.Decoder.Weights[0],
		)
	}

	if optimizer.StepCount != 1 {
		t.Fatalf(
			"optimizer was not restored: got %d",
			optimizer.StepCount,
		)
	}
}
func TestTrainingCheckpointSaveLoad(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	optimizer, err := NewAdam(0.01)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	parameters := []*Parameter{
		{
			Name:   "decoder",
			Values: []float32{1, 2, 3},
			Gradients: []float32{
				0.1,
				0.2,
				0.3,
			},
		},
	}

	if err := optimizer.Step(parameters); err != nil {
		t.Fatal(err)
	}

	model.Decoder.Weights[0] = 123.456

	checkpoint, err := trainer.CreateCheckpoint(
		7,
		42,
	)
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(
		t.TempDir(),
		"checkpoint.json",
	)

	err = SaveTrainingCheckpoint(
		path,
		checkpoint,
	)
	if err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadTrainingCheckpoint(
		path,
	)
	if err != nil {
		t.Fatal(err)
	}

	if loaded.Version != 1 {
		t.Fatalf(
			"expected version 1, got %d",
			loaded.Version,
		)
	}

	if loaded.Epoch != 7 {
		t.Fatalf(
			"expected epoch 7, got %d",
			loaded.Epoch,
		)
	}

	if loaded.Step != 42 {
		t.Fatalf(
			"expected step 42, got %d",
			loaded.Step,
		)
	}

	if loaded.Model.DecoderWeights[0] != 123.456 {
		t.Fatalf(
			"decoder weight mismatch: got %f",
			loaded.Model.DecoderWeights[0],
		)
	}

	if loaded.Optimizer.StepCount != 1 {
		t.Fatalf(
			"optimizer step mismatch: got %d",
			loaded.Optimizer.StepCount,
		)
	}
}
func TestTrainerResumeFromCheckpoint(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	for i := 0; i < 4; i++ {
		features := NewFeatureMatrix(20, 80)

		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				features.Set(
					row,
					col,
					float32((row+1)*(col+1))*0.0001,
				)
			}
		}

		labels, err := tokenizer.Encode("hel")
		if err != nil {
			t.Fatal(err)
		}

		if err := dataset.Add(
			TrainingSample{
				Features: features,
				Labels:   labels,
				Text:     "hel",
			},
		); err != nil {
			t.Fatal(err)
		}
	}

	model := NewSTTModel(
		80,
		8,
		vocabulary,
	)

	if model == nil {
		t.Fatal("expected model")
	}

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(
		model,
		optimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	loader, err := NewDataLoader(
		dataset,
		4,
		false,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	// First training phase.
	loss1, err := trainer.TrainEpoch(loader)
	if err != nil {
		t.Fatal(err)
	}

	stepBeforeCheckpoint := optimizer.StepCount

	checkpoint, err := trainer.CreateCheckpoint(
		1,
		stepBeforeCheckpoint,
	)
	if err != nil {
		t.Fatal(err)
	}

	if checkpoint.Optimizer.StepCount != stepBeforeCheckpoint {
		t.Fatalf(
			"checkpoint step mismatch: expected %d, got %d",
			stepBeforeCheckpoint,
			checkpoint.Optimizer.StepCount,
		)
	}

	// Destroy current model/optimizer state.
	for i := range model.Decoder.Weights {
		model.Decoder.Weights[i] = 999
	}

	optimizer.StepCount = 999

	// Restore checkpoint.
	if err := trainer.RestoreCheckpoint(
		checkpoint,
	); err != nil {
		t.Fatal(err)
	}

	if optimizer.StepCount != stepBeforeCheckpoint {
		t.Fatalf(
			"optimizer step was not restored: expected %d, got %d",
			stepBeforeCheckpoint,
			optimizer.StepCount,
		)
	}

	// Continue training.
	loss2, err := trainer.TrainEpoch(loader)
	if err != nil {
		t.Fatal(err)
	}

	if optimizer.StepCount <= stepBeforeCheckpoint {
		t.Fatalf(
			"optimizer did not continue: before=%d after=%d",
			stepBeforeCheckpoint,
			optimizer.StepCount,
		)
	}

	if loss2 >= loss1 {
		t.Fatalf(
			"expected resumed training loss to decrease: before=%f after=%f",
			loss1,
			loss2,
		)
	}

	t.Logf(
		"Initial loss: %.6f",
		loss1,
	)

	t.Logf(
		"Resumed loss: %.6f",
		loss2,
	)

	t.Logf(
		"Optimizer step: %d",
		optimizer.StepCount,
	)
}
func TestTrainerSaveLoadCheckpoint(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	features := NewFeatureMatrix(20, 80)
	for row := 0; row < 20; row++ {
		for col := 0; col < 80; col++ {
			features.Set(
				row,
				col,
				float32((row+1)*(col+1))*0.0001,
			)
		}
	}

	labels, err := tokenizer.Encode("hel")
	if err != nil {
		t.Fatal(err)
	}

	if err := dataset.Add(TrainingSample{
		Features: features,
		Labels:   labels,
		Text:     "hel",
	}); err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(80, 8, vocabulary)
	if model == nil {
		t.Fatal("expected model")
	}

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(model, optimizer)
	if err != nil {
		t.Fatal(err)
	}

	loader, err := NewDataLoader(dataset, 1, false, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = trainer.TrainEpoch(loader)
	if err != nil {
		t.Fatal(err)
	}

	stepBefore := optimizer.StepCount

	path := filepath.Join(t.TempDir(), "checkpoint.json")

	if err := trainer.SaveCheckpoint(path, 3, stepBefore); err != nil {
		t.Fatal(err)
	}

	for i := range model.Decoder.Weights {
		model.Decoder.Weights[i] = 999
	}

	optimizer.StepCount = 999

	if err := trainer.LoadCheckpoint(path); err != nil {
		t.Fatal(err)
	}

	if optimizer.StepCount != stepBefore {
		t.Fatalf(
			"optimizer step mismatch: expected %d, got %d",
			stepBefore,
			optimizer.StepCount,
		)
	}

	if model.Decoder.Weights[0] == 999 {
		t.Fatal("model weights were not restored")
	}

	t.Logf("Checkpoint restored: step=%d", optimizer.StepCount)
}
func TestTrainerAutoCheckpoint(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	for i := 0; i < 4; i++ {
		features := NewFeatureMatrix(20, 80)

		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				features.Set(
					row,
					col,
					float32((row+1)*(col+1))*0.0001,
				)
			}
		}

		labels, err := tokenizer.Encode("hel")
		if err != nil {
			t.Fatal(err)
		}

		if err := dataset.Add(TrainingSample{
			Features: features,
			Labels:   labels,
			Text:     "hel",
		}); err != nil {
			t.Fatal(err)
		}
	}

	model := NewSTTModel(80, 8, vocabulary)
	if model == nil {
		t.Fatal("expected model")
	}

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(model, optimizer)
	if err != nil {
		t.Fatal(err)
	}

	checkpointPath := filepath.Join(
		t.TempDir(),
		"auto-checkpoint.json",
	)

	config := TrainingConfig{
		Epochs:                4,
		BatchSize:             4,
		EarlyStoppingPatience: 0,
		EarlyStoppingMinDelta: 0,
		CheckpointPath:        checkpointPath,
		CheckpointEvery:       2,
	}

	history, err := trainer.TrainWithValidation(
		dataset,
		dataset,
		config,
	)
	if err != nil {
		t.Fatal(err)
	}

	if history == nil {
		t.Fatal("expected training history")
	}

	checkpoint, err := LoadTrainingCheckpoint(checkpointPath)
	if err != nil {
		t.Fatal(err)
	}

	if checkpoint.Epoch != 4 {
		t.Fatalf(
			"expected checkpoint epoch 4, got %d",
			checkpoint.Epoch,
		)
	}

	if checkpoint.Step <= 0 {
		t.Fatalf(
			"expected checkpoint step > 0, got %d",
			checkpoint.Step,
		)
	}

	t.Logf(
		"Auto checkpoint: epoch=%d step=%d",
		checkpoint.Epoch,
		checkpoint.Step,
	)
}
func TestTrainerResumeWithValidation(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	tokenizer, err := NewTokenizer(vocabulary)
	if err != nil {
		t.Fatal(err)
	}

	dataset := NewDataset()

	for i := 0; i < 4; i++ {
		features := NewFeatureMatrix(20, 80)

		for row := 0; row < 20; row++ {
			for col := 0; col < 80; col++ {
				features.Set(
					row,
					col,
					float32((row+1)*(col+1))*0.0001,
				)
			}
		}

		labels, err := tokenizer.Encode("hel")
		if err != nil {
			t.Fatal(err)
		}

		if err := dataset.Add(TrainingSample{
			Features: features,
			Labels:   labels,
			Text:     "hel",
		}); err != nil {
			t.Fatal(err)
		}
	}

	model := NewSTTModel(80, 8, vocabulary)
	if model == nil {
		t.Fatal("expected model")
	}

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(model, optimizer)
	if err != nil {
		t.Fatal(err)
	}

	checkpointPath := filepath.Join(
		t.TempDir(),
		"resume-checkpoint.json",
	)

	config := TrainingConfig{
		Epochs:                2,
		BatchSize:             4,
		CheckpointPath:        checkpointPath,
		CheckpointEvery:       2,
		EarlyStoppingPatience: 3,
		EarlyStoppingMinDelta: 0.1,
	}

	history, err := trainer.TrainWithValidation(
		dataset,
		dataset,
		config,
	)
	if err != nil {
		t.Fatal(err)
	}

	if len(history.TrainLoss) != 2 {
		t.Fatalf(
			"expected 2 epochs, got %d",
			len(history.TrainLoss),
		)
	}

	checkpoint, err := LoadTrainingCheckpoint(checkpointPath)
	if err != nil {
		t.Fatal(err)
	}
	if checkpoint.Version != 2 {
		t.Fatalf(
			"expected checkpoint version 2, got %d",
			checkpoint.Version,
		)
	}

	if checkpoint.BestModel == nil {
		t.Fatal("expected best model in checkpoint")
	}

	if checkpoint.EarlyStopping == nil {
		t.Fatal("expected early stopping state in checkpoint")
	}

	if checkpoint.BestEpoch < 0 {
		t.Fatalf(
			"expected valid best epoch, got %d",
			checkpoint.BestEpoch,
		)
	}

	if checkpoint.BestLoss <= 0 {
		t.Fatalf(
			"expected positive best loss, got %f",
			checkpoint.BestLoss,
		)
	}
	if checkpoint.Epoch != 2 {
		t.Fatalf(
			"expected checkpoint epoch 2, got %d",
			checkpoint.Epoch,
		)
	}

	stepBeforeResume := optimizer.StepCount

	resumeConfig := TrainingConfig{
		Epochs:                4,
		BatchSize:             4,
		CheckpointPath:        checkpointPath,
		CheckpointEvery:       2,
		EarlyStoppingPatience: 3,
		EarlyStoppingMinDelta: 0.1,
	}

	resumedHistory, err := trainer.ResumeWithValidation(
		checkpointPath,
		dataset,
		dataset,
		resumeConfig,
	)
	if err != nil {
		t.Fatal(err)
	}
	if resumedHistory == nil {
		t.Fatal("expected training history")
	}

	if resumedHistory.BestEpoch < checkpoint.BestEpoch {
		t.Fatalf(
			"resumed best epoch went backwards: checkpoint=%d resumed=%d",
			checkpoint.BestEpoch,
			resumedHistory.BestEpoch,
		)
	}

	if resumedHistory.BestLoss > checkpoint.BestLoss {
		t.Fatalf(
			"resumed best loss got worse: checkpoint=%f resumed=%f",
			checkpoint.BestLoss,
			resumedHistory.BestLoss,
		)
	}
	if len(resumedHistory.TrainLoss) != 2 {
		t.Fatalf(
			"expected 2 resumed epochs, got %d",
			len(resumedHistory.TrainLoss),
		)
	}

	if optimizer.StepCount <= stepBeforeResume {
		t.Fatalf(
			"optimizer did not continue: before=%d after=%d",
			stepBeforeResume,
			optimizer.StepCount,
		)
	}

	finalCheckpoint, err := LoadTrainingCheckpoint(checkpointPath)
	if err != nil {
		t.Fatal(err)
	}

	if finalCheckpoint.Epoch != 4 {
		t.Fatalf(
			"expected final checkpoint epoch 4, got %d",
			finalCheckpoint.Epoch,
		)
	}

	t.Logf(
		"Resume successful: epoch=%d step=%d",
		finalCheckpoint.Epoch,
		finalCheckpoint.Step,
	)
}
func TestTrainingCheckpointTrainingState(t *testing.T) {
	earlyStopping, err := NewEarlyStopping(3, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := earlyStopping.Update(10); err != nil {
		t.Fatal(err)
	}

	if _, err := earlyStopping.Update(10.05); err != nil {
		t.Fatal(err)
	}

	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(80, 8, vocabulary)
	if model == nil {
		t.Fatal("expected model")
	}

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(model, optimizer)
	if err != nil {
		t.Fatal(err)
	}

	bestSnapshot, err := model.Snapshot()
	if err != nil {
		t.Fatal(err)
	}

	checkpoint, err := trainer.CreateTrainingCheckpoint(
		5,
		12,
		bestSnapshot,
		3.25,
		4,
		earlyStopping,
	)
	if err != nil {
		t.Fatal(err)
	}

	if checkpoint.Version != 2 {
		t.Fatalf("expected version 2, got %d", checkpoint.Version)
	}

	if checkpoint.Epoch != 5 {
		t.Fatalf("expected epoch 5, got %d", checkpoint.Epoch)
	}

	if checkpoint.Step != 12 {
		t.Fatalf("expected step 12, got %d", checkpoint.Step)
	}

	if checkpoint.BestModel == nil {
		t.Fatal("expected best model")
	}

	if checkpoint.BestEpoch != 4 {
		t.Fatalf("expected best epoch 4, got %d", checkpoint.BestEpoch)
	}

	if checkpoint.BestLoss != 3.25 {
		t.Fatalf("expected best loss 3.25, got %f", checkpoint.BestLoss)
	}

	if checkpoint.EarlyStopping == nil {
		t.Fatal("expected early stopping state")
	}

	if checkpoint.EarlyStopping.WaitCount != 1 {
		t.Fatalf(
			"expected early stopping wait count 1, got %d",
			checkpoint.EarlyStopping.WaitCount,
		)
	}

	t.Logf(
		"Training state: epoch=%d step=%d bestEpoch=%d bestLoss=%.2f wait=%d",
		checkpoint.Epoch,
		checkpoint.Step,
		checkpoint.BestEpoch,
		checkpoint.BestLoss,
		checkpoint.EarlyStopping.WaitCount,
	)
}
func TestEarlyStoppingSnapshotRestore(t *testing.T) {
	earlyStopping, err := NewEarlyStopping(3, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := earlyStopping.Update(10); err != nil {
		t.Fatal(err)
	}

	if _, err := earlyStopping.Update(10.05); err != nil {
		t.Fatal(err)
	}

	state, err := earlyStopping.Snapshot()
	if err != nil {
		t.Fatal(err)
	}

	restored, err := NewEarlyStopping(3, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	if err := restored.Restore(state); err != nil {
		t.Fatal(err)
	}

	if restored.BestLoss() != earlyStopping.BestLoss() {
		t.Fatalf(
			"best loss mismatch: expected %f got %f",
			earlyStopping.BestLoss(),
			restored.BestLoss(),
		)
	}

	if restored.WaitCount() != earlyStopping.WaitCount() {
		t.Fatalf(
			"wait count mismatch: expected %d got %d",
			earlyStopping.WaitCount(),
			restored.WaitCount(),
		)
	}

	t.Logf(
		"EarlyStopping restored: best=%.6f wait=%d",
		restored.BestLoss(),
		restored.WaitCount(),
	)
}
func TestTrainerResumeRestoresTrainingState(t *testing.T) {
	vocabulary, err := NewEnglishVocabulary()
	if err != nil {
		t.Fatal(err)
	}

	model := NewSTTModel(80, 8, vocabulary)
	if model == nil {
		t.Fatal("expected model")
	}

	optimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	trainer, err := NewTrainer(model, optimizer)
	if err != nil {
		t.Fatal(err)
	}
	parameter, err := NewParameter(
		"test.parameter",
		[]float32{1},
		[]float32{0.1},
	)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 12; i++ {
		if err := optimizer.Step([]*Parameter{
			parameter,
		}); err != nil {
			t.Fatal(err)
		}
	}

	if optimizer.StepCount != 12 {
		t.Fatalf(
			"expected optimizer step 12, got %d",
			optimizer.StepCount,
		)
	}

	earlyStopping, err := NewEarlyStopping(3, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := earlyStopping.Update(10); err != nil {
		t.Fatal(err)
	}

	if _, err := earlyStopping.Update(10.05); err != nil {
		t.Fatal(err)
	}

	bestModel, err := model.Snapshot()
	if err != nil {
		t.Fatal(err)
	}

	checkpoint, err := trainer.CreateTrainingCheckpoint(
		5,
		12,
		bestModel,
		3.25,
		4,
		earlyStopping,
	)
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(
		t.TempDir(),
		"resume_training_state.json",
	)

	if err := SaveTrainingCheckpoint(path, checkpoint); err != nil {
		t.Fatal(err)
	}

	loadedCheckpoint, err := LoadTrainingCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}

	if loadedCheckpoint.BestModel == nil {
		t.Fatal("expected best model in checkpoint")
	}

	if loadedCheckpoint.EarlyStopping == nil {
		t.Fatal("expected early stopping state in checkpoint")
	}

	if loadedCheckpoint.BestEpoch != 4 {
		t.Fatalf(
			"expected best epoch 4, got %d",
			loadedCheckpoint.BestEpoch,
		)
	}

	if loadedCheckpoint.BestLoss != 3.25 {
		t.Fatalf(
			"expected best loss 3.25, got %f",
			loadedCheckpoint.BestLoss,
		)
	}

	if loadedCheckpoint.EarlyStopping.WaitCount != 1 {
		t.Fatalf(
			"expected wait count 1, got %d",
			loadedCheckpoint.EarlyStopping.WaitCount,
		)
	}

	resumeModel := NewSTTModel(80, 8, vocabulary)
	if resumeModel == nil {
		t.Fatal("expected resume model")
	}

	resumeOptimizer, err := NewAdam(0.02)
	if err != nil {
		t.Fatal(err)
	}

	resumeTrainer, err := NewTrainer(
		resumeModel,
		resumeOptimizer,
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := resumeTrainer.RestoreCheckpoint(
		loadedCheckpoint,
	); err != nil {
		t.Fatal(err)
	}

	if resumeOptimizer.StepCount != 12 {
		t.Fatalf(
			"expected optimizer step 12, got %d",
			resumeOptimizer.StepCount,
		)
	}

	restoredModel, err := resumeModel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}

	if len(restoredModel.DecoderWeights) !=
		len(loadedCheckpoint.Model.DecoderWeights) {
		t.Fatal("decoder weight size mismatch")
	}

	for i := range restoredModel.DecoderWeights {
		if restoredModel.DecoderWeights[i] !=
			loadedCheckpoint.Model.DecoderWeights[i] {
			t.Fatalf(
				"decoder weight mismatch at index %d",
				i,
			)
		}
	}

	resumedEarlyStopping, err := NewEarlyStopping(3, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	if err := resumedEarlyStopping.Restore(
		loadedCheckpoint.EarlyStopping,
	); err != nil {
		t.Fatal(err)
	}

	if resumedEarlyStopping.BestLoss() != 10 {
		t.Fatalf(
			"expected restored early stopping best loss 10, got %f",
			resumedEarlyStopping.BestLoss(),
		)
	}

	if resumedEarlyStopping.WaitCount() != 1 {
		t.Fatalf(
			"expected restored wait count 1, got %d",
			resumedEarlyStopping.WaitCount(),
		)
	}

	t.Logf(
		"Training state restored: epoch=%d step=%d bestEpoch=%d bestLoss=%.2f wait=%d",
		loadedCheckpoint.Epoch,
		loadedCheckpoint.Step,
		loadedCheckpoint.BestEpoch,
		loadedCheckpoint.BestLoss,
		loadedCheckpoint.EarlyStopping.WaitCount,
	)
}
func TestCommonVoiceEntry(t *testing.T) {
	entry := CommonVoiceEntry{
		AudioPath: "clips/test.mp3",
		Text:      "hello world",
	}

	if entry.AudioPath == "" {
		t.Fatal("audio path is empty")
	}

	if entry.Text == "" {
		t.Fatal("text is empty")
	}
}
func TestLoadCommonVoiceManifest(t *testing.T) {
	root := filepath.Join(
		"..",
		"..",
		"data",
		"raw",
		"commonvoice",
		"fa",
	)

	manifestPath := filepath.Join(root, "train.tsv")

	entries, err := LoadCommonVoiceManifest(manifestPath)
	if err != nil {
		t.Fatalf("LoadCommonVoiceManifest failed: %v", err)
	}

	if len(entries) == 0 {
		t.Fatal("manifest returned no entries")
	}

	t.Logf("loaded %d Common Voice entries", len(entries))

	t.Logf(
		"first entry: path=%s text=%s",
		entries[0].AudioPath,
		entries[0].Text,
	)

	if entries[0].AudioPath == "" {
		t.Fatal("first audio path is empty")
	}

	if entries[0].Text == "" {
		t.Fatal("first text is empty")
	}
}
func TestCommonVoiceImporterImport(t *testing.T) {
	root := filepath.Join(
		"..",
		"..",
		"data",
		"raw",
		"commonvoice",
		"fa",
	)

	manifestPath := filepath.Join(root, "train.tsv")

	entries, err := LoadCommonVoiceManifest(manifestPath)
	if err != nil {
		t.Fatalf("LoadCommonVoiceManifest failed: %v", err)
	}

	texts := make([]string, 0, len(entries))

	for _, entry := range entries {
		texts = append(texts, entry.Text)
	}

	tokenizer, err := NewTokenizerFromTexts(texts)
	if err != nil {
		t.Fatalf("NewTokenizerFromTexts failed: %v", err)
	}

	ffmpeg, err := media.NewFFmpeg("")
	if err != nil {
		t.Fatalf("NewFFmpeg failed: %v", err)
	}

	importer, err := NewCommonVoiceImporter(
		ffmpeg,
		DefaultFeatureConfig(),
		tokenizer,
	)
	if err != nil {
		t.Fatalf("NewCommonVoiceImporter failed: %v", err)
	}

	dataset, err := importer.Import(
		context.Background(),
		root,
		manifestPath,
		1,
	)
	if err != nil {
		t.Fatalf("Import failed: %v", err)
	}

	if dataset.Len() != 1 {
		t.Fatalf(
			"expected 1 sample, got %d",
			dataset.Len(),
		)
	}

	sample, err := dataset.Get(0)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}

	if sample.Features == nil {
		t.Fatal("features are nil")
	}

	if len(sample.Labels) == 0 {
		t.Fatal("labels are empty")
	}

	if sample.Text == "" {
		t.Fatal("text is empty")
	}

	t.Logf("text: %s", sample.Text)
	t.Logf("labels: %d", len(sample.Labels))
	t.Logf(
		"features: time=%d dim=%d",
		sample.Features.Rows,
		sample.Features.Cols,
	)
}
func TestSplitCommonVoice(t *testing.T) {
	entries := make([]CommonVoiceEntry, 100)

	for i := range entries {
		entries[i] = CommonVoiceEntry{
			AudioPath: "audio.mp3",
			Text:      "متن",
		}
	}

	split, err := SplitCommonVoice(
		entries,
		0.1,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatalf("SplitCommonVoice failed: %v", err)
	}

	if len(split.Train) != 90 {
		t.Fatalf(
			"expected 90 train entries, got %d",
			len(split.Train),
		)
	}

	if len(split.Valid) != 10 {
		t.Fatalf(
			"expected 10 validation entries, got %d",
			len(split.Valid),
		)
	}
}
func TestCommonVoiceSplitAndImport(t *testing.T) {
	root := filepath.Join(
		"..",
		"..",
		"data",
		"raw",
		"commonvoice",
		"fa",
	)

	manifestPath := filepath.Join(root, "train.tsv")

	entries, err := LoadCommonVoiceManifest(manifestPath)
	if err != nil {
		t.Fatalf("LoadCommonVoiceManifest failed: %v", err)
	}

	split, err := SplitCommonVoice(
		entries,
		0.1,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatalf("SplitCommonVoice failed: %v", err)
	}

	texts := make([]string, 0, len(split.Train))

	for _, entry := range split.Train {
		texts = append(texts, entry.Text)
	}

	tokenizer, err := NewTokenizerFromTexts(texts)
	if err != nil {
		t.Fatalf("NewTokenizerFromTexts failed: %v", err)
	}

	ffmpeg, err := media.NewFFmpeg("")
	if err != nil {
		t.Fatalf("NewFFmpeg failed: %v", err)
	}

	importer, err := NewCommonVoiceImporter(
		ffmpeg,
		DefaultFeatureConfig(),
		tokenizer,
	)
	if err != nil {
		t.Fatalf("NewCommonVoiceImporter failed: %v", err)
	}

	trainDataset, err := importer.ImportEntries(
		context.Background(),
		root,
		split.Train,
		4,
	)
	if err != nil {
		t.Fatalf("train import failed: %v", err)
	}

	validDataset, err := importer.ImportEntries(
		context.Background(),
		root,
		split.Valid,
		2,
	)
	if err != nil {
		t.Fatalf("validation import failed: %v", err)
	}

	if trainDataset.Len() != 4 {
		t.Fatalf(
			"expected 4 train samples, got %d",
			trainDataset.Len(),
		)
	}

	if validDataset.Len() != 2 {
		t.Fatalf(
			"expected 2 validation samples, got %d",
			validDataset.Len(),
		)
	}

	t.Logf(
		"train samples: %d",
		trainDataset.Len(),
	)

	t.Logf(
		"validation samples: %d",
		validDataset.Len(),
	)
}
func TestCommonVoiceCreateBatch(t *testing.T) {
	root := filepath.Join(
		"..",
		"..",
		"data",
		"raw",
		"commonvoice",
		"fa",
	)

	manifestPath := filepath.Join(root, "train.tsv")

	entries, err := LoadCommonVoiceManifest(manifestPath)
	if err != nil {
		t.Fatalf("LoadCommonVoiceManifest failed: %v", err)
	}

	split, err := SplitCommonVoice(
		entries,
		0.1,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatalf("SplitCommonVoice failed: %v", err)
	}

	texts := make([]string, 0, len(split.Train))

	for _, entry := range split.Train {
		texts = append(texts, entry.Text)
	}

	tokenizer, err := NewTokenizerFromTexts(texts)
	if err != nil {
		t.Fatalf("NewTokenizerFromTexts failed: %v", err)
	}

	ffmpeg, err := media.NewFFmpeg("")
	if err != nil {
		t.Fatalf("NewFFmpeg failed: %v", err)
	}

	importer, err := NewCommonVoiceImporter(
		ffmpeg,
		DefaultFeatureConfig(),
		tokenizer,
	)
	if err != nil {
		t.Fatalf("NewCommonVoiceImporter failed: %v", err)
	}

	dataset, err := importer.ImportEntries(
		context.Background(),
		root,
		split.Train,
		4,
	)
	if err != nil {
		t.Fatalf("ImportEntries failed: %v", err)
	}

	batch, err := CreateBatch(
		dataset,
		0,
		dataset.Len(),
	)
	if err != nil {
		t.Fatalf("CreateBatch failed: %v", err)
	}

	if batch.BatchSize != 4 {
		t.Fatalf(
			"expected batch size 4, got %d",
			batch.BatchSize,
		)
	}

	if batch.FeatureDim != 80 {
		t.Fatalf(
			"expected feature dim 80, got %d",
			batch.FeatureDim,
		)
	}

	if batch.TimeSteps <= 0 {
		t.Fatal("batch time steps must be positive")
	}

	if len(batch.InputLengths) != 4 {
		t.Fatalf(
			"expected 4 input lengths, got %d",
			len(batch.InputLengths),
		)
	}

	if len(batch.LabelLengths) != 4 {
		t.Fatalf(
			"expected 4 label lengths, got %d",
			len(batch.LabelLengths),
		)
	}

	if len(batch.Labels) == 0 {
		t.Fatal("batch labels are empty")
	}

	t.Logf(
		"batch size=%d time=%d feature_dim=%d",
		batch.BatchSize,
		batch.TimeSteps,
		batch.FeatureDim,
	)

	t.Logf(
		"input lengths=%v",
		batch.InputLengths,
	)

	t.Logf(
		"label lengths=%v",
		batch.LabelLengths,
	)

	t.Logf(
		"total labels=%d",
		len(batch.Labels),
	)
}
func TestCommonVoiceModelForward(t *testing.T) {
	root := filepath.Join(
		"..",
		"..",
		"data",
		"raw",
		"commonvoice",
		"fa",
	)

	manifestPath := filepath.Join(root, "train.tsv")

	entries, err := LoadCommonVoiceManifest(manifestPath)
	if err != nil {
		t.Fatalf("LoadCommonVoiceManifest failed: %v", err)
	}

	split, err := SplitCommonVoice(
		entries,
		0.1,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatalf("SplitCommonVoice failed: %v", err)
	}

	texts := make([]string, 0, len(split.Train))

	for _, entry := range split.Train {
		texts = append(texts, entry.Text)
	}

	tokenizer, err := NewTokenizerFromTexts(texts)
	if err != nil {
		t.Fatalf("NewTokenizerFromTexts failed: %v", err)
	}

	ffmpeg, err := media.NewFFmpeg("")
	if err != nil {
		t.Fatalf("NewFFmpeg failed: %v", err)
	}

	importer, err := NewCommonVoiceImporter(
		ffmpeg,
		DefaultFeatureConfig(),
		tokenizer,
	)
	if err != nil {
		t.Fatalf("NewCommonVoiceImporter failed: %v", err)
	}

	dataset, err := importer.ImportEntries(
		context.Background(),
		root,
		split.Train,
		4,
	)
	if err != nil {
		t.Fatalf("ImportEntries failed: %v", err)
	}

	batch, err := CreateBatch(
		dataset,
		0,
		dataset.Len(),
	)
	if err != nil {
		t.Fatalf("CreateBatch failed: %v", err)
	}

	if batch.BatchSize != 4 {
		t.Fatalf(
			"expected batch size 4, got %d",
			batch.BatchSize,
		)
	}

	if batch.FeatureDim != 80 {
		t.Fatalf(
			"expected feature dim 80, got %d",
			batch.FeatureDim,
		)
	}

	if batch.TimeSteps <= 0 {
		t.Fatal("batch time steps must be positive")
	}

	if len(batch.InputLengths) != 4 {
		t.Fatalf(
			"expected 4 input lengths, got %d",
			len(batch.InputLengths),
		)
	}

	if len(batch.LabelLengths) != 4 {
		t.Fatalf(
			"expected 4 label lengths, got %d",
			len(batch.LabelLengths),
		)
	}

	if len(batch.Labels) == 0 {
		t.Fatal("batch labels are empty")
	}

	input := batchToTensor3D(batch)

	if input == nil {
		t.Fatal("failed to convert batch to Tensor3D")
	}

	if input.Batch != batch.BatchSize {
		t.Fatalf(
			"unexpected tensor batch: got %d want %d",
			input.Batch,
			batch.BatchSize,
		)
	}

	if input.Time != batch.TimeSteps {
		t.Fatalf(
			"unexpected tensor time: got %d want %d",
			input.Time,
			batch.TimeSteps,
		)
	}

	if input.Features != batch.FeatureDim {
		t.Fatalf(
			"unexpected tensor features: got %d want %d",
			input.Features,
			batch.FeatureDim,
		)
	}

	model := NewSTTModel(
		batch.FeatureDim,
		64,
		tokenizer.vocabulary,
	)

	if model == nil {
		t.Fatal("failed to create STT model")
	}

	logits, err := model.Forward(
		input,
		batch.InputLengths,
	)

	if err != nil {
		t.Fatalf(
			"model forward failed: %v",
			err,
		)
	}

	if logits == nil {
		t.Fatal("logits is nil")
	}

	t.Logf(
		"input: batch=%d time=%d features=%d",
		input.Batch,
		input.Time,
		input.Features,
	)

	t.Logf(
		"logits: batch=%d time=%d features=%d",
		logits.Batch,
		logits.Time,
		logits.Features,
	)

	if logits.Batch != batch.BatchSize {
		t.Fatalf(
			"unexpected logits batch: got %d want %d",
			logits.Batch,
			batch.BatchSize,
		)
	}

	if logits.Time != batch.TimeSteps {
		t.Fatalf(
			"unexpected logits time: got %d want %d",
			logits.Time,
			batch.TimeSteps,
		)
	}

	if logits.Features != tokenizer.vocabulary.Size() {
		t.Fatalf(
			"unexpected logits vocabulary: got %d want %d",
			logits.Features,
			tokenizer.vocabulary.Size(),
		)
	}
	t.Logf(
		"batch size=%d time=%d feature_dim=%d",
		batch.BatchSize,
		batch.TimeSteps,
		batch.FeatureDim,
	)

	t.Logf(
		"input lengths=%v",
		batch.InputLengths,
	)

	t.Logf(
		"label lengths=%v",
		batch.LabelLengths,
	)

	t.Logf(
		"total labels=%d",
		len(batch.Labels),
	)
	loss, err := CTCLossBatch(
		logits,
		batch.Labels,
		batch.InputLengths,
		batch.LabelLengths,
		tokenizer.vocabulary.BlankID(),
	)

	if err != nil {
		t.Fatalf(
			"CTCLossBatch failed: %v",
			err,
		)
	}

	if math.IsNaN(float64(loss)) ||
		math.IsInf(float64(loss), 0) {
		t.Fatalf(
			"CTC loss is not finite: %v",
			loss,
		)
	}

	if loss < 0 {
		t.Fatalf(
			"CTC loss must not be negative: %v",
			loss,
		)
	}

	t.Logf(
		"real CTC loss=%f",
		loss,
	)
	optimizer, err := NewAdam(1e-4)
	if err != nil {
		t.Fatalf(
			"NewAdam failed: %v",
			err,
		)
	}

	trainer := &Trainer{
		Model:     model,
		Optimizer: optimizer,
		RNG:       rand.New(rand.NewSource(42)),
	}

	lossBefore := loss

	trainLoss, err := trainer.TrainBatch(
		batch,
		batch.Labels,
	)
	if err != nil {
		t.Fatalf(
			"TrainBatch failed: %v",
			err,
		)
	}

	if math.IsNaN(float64(trainLoss)) ||
		math.IsInf(float64(trainLoss), 0) {
		t.Fatalf(
			"train loss is not finite: %v",
			trainLoss,
		)
	}

	inputAfter, err := batch.ToTensor()
	if err != nil {
		t.Fatalf(
			"batch.ToTensor after training failed: %v",
			err,
		)
	}

	logitsAfter, err := model.Forward(
		inputAfter,
		batch.InputLengths,
	)
	if err != nil {
		t.Fatalf(
			"forward after training failed: %v",
			err,
		)
	}

	lossAfter, err := CTCLossBatch(
		logitsAfter,
		batch.Labels,
		batch.InputLengths,
		batch.LabelLengths,
		tokenizer.vocabulary.BlankID(),
	)
	if err != nil {
		t.Fatalf(
			"CTCLossBatch after training failed: %v",
			err,
		)
	}

	if math.IsNaN(float64(lossAfter)) ||
		math.IsInf(float64(lossAfter), 0) {
		t.Fatalf(
			"loss after training is not finite: %v",
			lossAfter,
		)
	}

	t.Logf(
		"CTC loss before=%f",
		lossBefore,
	)

	t.Logf(
		"TrainBatch loss=%f",
		trainLoss,
	)

	t.Logf(
		"CTC loss after=%f",
		lossAfter,
	)
}
func TestCommonVoiceTrainingLoop(t *testing.T) {
	root := filepath.Join(
		"..",
		"..",
		"data",
		"raw",
		"commonvoice",
		"fa",
	)

	manifestPath := filepath.Join(
		root,
		"train.tsv",
	)

	entries, err := LoadCommonVoiceManifest(
		manifestPath,
	)
	if err != nil {
		t.Fatalf(
			"LoadCommonVoiceManifest failed: %v",
			err,
		)
	}

	split, err := SplitCommonVoice(
		entries,
		0.1,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatalf(
			"SplitCommonVoice failed: %v",
			err,
		)
	}

	// Vocabulary فقط از train ساخته می‌شود.
	texts := make(
		[]string,
		0,
		len(split.Train),
	)

	for _, entry := range split.Train {
		texts = append(
			texts,
			entry.Text,
		)
	}

	tokenizer, err := NewTokenizerFromTexts(
		texts,
	)
	if err != nil {
		t.Fatalf(
			"NewTokenizerFromTexts failed: %v",
			err,
		)
	}

	ffmpeg, err := media.NewFFmpeg("")
	if err != nil {
		t.Fatalf(
			"NewFFmpeg failed: %v",
			err,
		)
	}

	importer, err := NewCommonVoiceImporter(
		ffmpeg,
		DefaultFeatureConfig(),
		tokenizer,
	)
	if err != nil {
		t.Fatalf(
			"NewCommonVoiceImporter failed: %v",
			err,
		)
	}

	// فقط 50 sample واقعی.
	dataset, err := importer.ImportEntries(
		context.Background(),
		root,
		split.Train,
		8,
	)
	if err != nil {
		t.Fatalf(
			"ImportEntries failed: %v",
			err,
		)
	}

	if dataset.Len() != 8 {
		t.Fatalf(
			"expected 50 samples, got %d",
			dataset.Len(),
		)
	}

	loader, err := NewDataLoader(
		dataset,
		2,
		true,
		rand.New(rand.NewSource(42)),
	)
	if err != nil {
		t.Fatalf(
			"NewDataLoader failed: %v",
			err,
		)
	}

	model := NewSTTModel(
		DefaultFeatureConfig().MelBins,
		32,
		tokenizer.vocabulary,
	)
	if model == nil {
		t.Fatal("failed to create STT model")
	}

	optimizer, err := NewAdam(1e-4)
	if err != nil {
		t.Fatalf(
			"NewAdam failed: %v",
			err,
		)
	}

	trainer := &Trainer{
		Model:     model,
		Optimizer: optimizer,
		RNG:       rand.New(rand.NewSource(42)),
	}

	step := 0

	for {
		batch, ok, err := loader.Next()
		if err != nil {
			t.Fatalf(
				"loader.Next failed: %v",
				err,
			)
		}

		if !ok {
			break
		}

		loss, err := trainer.TrainBatch(
			batch,
			batch.Labels,
		)
		if err != nil {
			t.Fatalf(
				"TrainBatch failed at step %d: %v",
				step,
				err,
			)
		}

		if math.IsNaN(float64(loss)) ||
			math.IsInf(float64(loss), 0) {
			t.Fatalf(
				"non-finite loss at step %d: %v",
				step,
				loss,
			)
		}

		if loss < 0 {
			t.Fatalf(
				"negative loss at step %d: %v",
				step,
				loss,
			)
		}

		if step%5 == 0 {
			t.Logf(
				"step=%d batch=%d time=%d loss=%f",
				step,
				batch.BatchSize,
				batch.TimeSteps,
				loss,
			)
		}

		step++
	}

	if step == 0 {
		t.Fatal("training loop executed zero steps")
	}

	t.Logf(
		"training completed: samples=%d steps=%d",
		dataset.Len(),
		step,
	)
}
