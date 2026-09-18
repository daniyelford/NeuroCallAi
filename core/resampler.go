package core

type LinearResampler struct{}

func (r *LinearResampler) Resample(
	input []int16,
	fromRate int,
	toRate int,
) []int16 {

	if fromRate == toRate {
		return input
	}

	if len(input) == 0 {
		return nil
	}

	ratio := float64(toRate) / float64(fromRate)

	outputLen := int(float64(len(input)) * ratio)

	output := make([]int16, outputLen)

	for i := 0; i < outputLen; i++ {

		src := int(float64(i) / ratio)

		if src >= len(input) {
			src = len(input) - 1
		}

		output[i] = input[src]
	}

	return output
}
