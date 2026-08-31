package core

import "github.com/daniyelford/NeuroCallAi/pkg/neurocall"

func NewCodecRegistry() *CodecRegistry {
	return &CodecRegistry{preferred: defaultCodecRegistry}
}
func (r *CodecRegistry) Negotiate(remote []neurocall.Codec) (neurocall.Codec, error) {
	for _, local := range r.preferred {
		for _, remoteCodec := range remote {
			if local.Name == remoteCodec.Name {
				return local, nil
			}
		}
	}
	return neurocall.Codec{}, neurocall.ErrNoCommonCodec
}
