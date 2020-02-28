package normalizer

import ()

type Normalizer interface {
	Normalize(normalized Normalized) error
}

type normalizer struct {
	Normalizer Normalizer
}

func newNormalizer() normalizer {
	return normalizer{
		Normalizer: NewDefaultNormalizer(),
	}
}

func (n normalizer) Normalize(normalized Normalized) error {

	return nil
}

type Option func(*normalizer)

func WithBertNormalizer() Option {
	return func(o *normalizer) {
		NewBertNormalizer()
	}
}

func NewNormalizer(opts ...Option) Normalizer {
	nml := newNormalizer()

	for _, o := range opts {
		o(&nml)
	}

	return nml
}
