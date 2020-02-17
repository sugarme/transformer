package normalizer

import (
// "fmt"
)

type Normalizer interface {
	Normalize()
}

type normalizer struct {
	Normalizer Normalizer
}


func newNormalizer() normalizer{
	return normalizer{
		Normalizer: NewBasicNormalizer()
	}
}


type Option func(*normalizer)

func WithBertNormalizer() Option {
	return func(o *normalizer) {
		Normalizer: NewBertNormalizer()
	}
}

func NewNormalizer(opts ...Option) Normalizer {
	nml := newNormalizer()

	for _, o := range opts{
		o(&nml)
	}

	return nml
}

