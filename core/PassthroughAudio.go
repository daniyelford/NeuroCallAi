package core

func (PassthroughAudio) Encode(
	pcm []int16,
) []byte {
	return PCM16ToBytes(pcm)
}
func (PassthroughAudio) Decode(
	data []byte,
) []int16 {
	return BytesToPCM16(data)
}
