package core

func (PCMA) Encode(pcm []int16) []byte {
	out := make([]byte, len(pcm))

	for i, sample := range pcm {
		out[i] = linearToALaw(sample)
	}

	return out
}
func (PCMA) Decode(data []byte) []int16 {
	out := make([]int16, len(data))

	for i, sample := range data {
		out[i] = aLawToLinear(sample)
	}

	return out
}
