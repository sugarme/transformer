package ann

import (
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ConsOpt is a construction option for layers
type ConsOpt func(Layer) (Layer, error)

type namesetter interface {
	SetName(a string) error
}

type sizeSetter interface {
	SetSize(int) error
}

type actSetter interface {
	SetActivationFn(act ActivationFunction) error
}

// WithName creates a layer that is named.
//
// If the layer is unnameable (i.e. trivial layers), then there is no effect.
func WithName(name string) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *FC:
			l.name = name
			return layer, nil
		case *layerNorm:
			l.name = name
			return layer, nil
		case *LSTM:
		case *Conv:
		case unnameable:
			return layer, nil
		case namesetter:
			err := l.SetName(name)
			return layer, err
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("WithName Unhandled Layer type: %T", layer)
	}
}

// AsBatched defines a layer that performs batched operation or not.
// By default most layers in this package are batched.
func AsBatched(batched bool) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *FC:
			l.batched = batched
			return layer, nil
		case *LSTM:
		case *Conv:
		case reshape:
			return layer, nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("AsBatched Unhandled Layer type: %T", layer)
	}
}

// WithBias defines a layer with or without a bias
func WithBias(withbias bool) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *FC:
			l.nobias = !withbias
			return layer, nil
		}
		return layer, nil
	}
}

// WithSize automatically creates a layer of the given sizes.
func WithSize(size ...int) ConsOpt {
	return func(layer Layer) (Layer, error) {
		// NO RESHAPE ALLOWED
		switch l := layer.(type) {
		case *FC:
			l.size = size[0]
			return l, nil
		case sizeSetter:
			l.SetSize(size[0])
			return layer, nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("WithSize Unhandled Layer type: %T", layer)
	}
}

// WithActivation sets the activation function of a layer
func WithActivation(act ActivationFunction) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *FC:
			l.act = act
			return layer, nil
		case reshape:
			return layer, nil
		case actSetter:
			l.SetActivationFn(act)
			return layer, nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("WithActivation Unhandled Layer type: %T", layer)
	}
}

// ToShape is a ConsOpt for Reshape only.
func ToShape(shp ...int) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch layer.(type) {
		case reshape:
			return reshape(tensor.Shape(shp)), nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("ToShape Unhandled Layer type: %T", layer)
	}
}

// WithProbability is a ConsOpt for Dropout only.
func WithProbability(prob float64) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch layer.(type) {
		case dropout:
			return dropout(prob), nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("WithProbability Unhandled Layer type: %T", layer)
	}
}

// WithEps is a ConsOpt for constructing Layer Norms only.
func WithEps(eps float64) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *layerNorm:
			l.eps = eps
			return l, nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("WithEps Unhandled Layer type: %T", layer)
	}
}

// WithConst is a construction option for the skip Layer
func WithConst(c *G.Node) ConsOpt {
	return func(l Layer) (Layer, error) {
		if a, ok := l.(*skip); ok {
			a.b = c
			return l, nil
		}
		return nil, errors.Errorf("WithConst expects a *skip. Got %T instead", l)
	}
}
