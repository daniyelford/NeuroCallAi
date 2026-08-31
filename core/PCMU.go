package core

func (PCMU) Encode(pcm []int16) []byte {
	out := make([]byte, len(pcm))

	for i, sample := range pcm {
		out[i] = linearToMuLaw(sample)
	}

	return out
}
func (PCMU) Decode(data []byte) []int16 {
	out := make([]int16, len(data))

	for i, sample := range data {
		out[i] = muLawToLinear(sample)
	}

	return out
}
