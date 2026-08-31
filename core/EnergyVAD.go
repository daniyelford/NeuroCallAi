package core

import (
	"context"

	"github.com/daniyelford/NeuroCallAi/pkg/neurocall"
)

func NewEnergyVAD(threshold int64) *EnergyVAD {
	if threshold <= 0 {
		threshold = 500
	}

	return &EnergyVAD{
		Threshold: threshold,
	}
}
func (v *EnergyVAD) Process(
	ctx context.Context,
	frame neurocall.AudioFrame,
) (bool, error) {

	select {
	case <-ctx.Done():
		return false, ctx.Err()

	default:
	}

	if len(frame.Data) == 0 {
		return false, nil
	}

	var energy int64

	for _, sample := range frame.Data {
		value := int64(sample)

		if value < 0 {
			value = -value
		}

		energy += value
	}

	energy /= int64(len(frame.Data))

	return energy >= v.Threshold, nil
}
