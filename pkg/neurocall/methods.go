package neurocall

func NewCodec(
	name string,
	payloadType uint8,
	clockRate int,
	channels int,
) Codec {
	return Codec{
		Name:        name,
		PayloadType: payloadType,
		ClockRate:   clockRate,
		Channels:    channels,
	}
}
func (c Codec) Valid() bool {
	return c.Name != "" && c.ClockRate > 0 && c.Channels > 0
}
func (m Message) Valid() bool {
	return m.Role != "" || m.Content != ""
}
