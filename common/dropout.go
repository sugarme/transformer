package common

import (
	ts "github.com/sugarme/gotch/tensor"
)

type Dropout struct {
	dropoutProb float64
}

func NewDropout(p float64) Dropout {
	return Dropout{
		dropoutProb: p,
	}
}

func (d Dropout) ForwardT(input ts.Tensor, train bool) (retVal ts.Tensor) {
	return ts.MustDropout(input, d.dropoutProb, train)
}
