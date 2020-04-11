package ann

import G "gorgonia.org/gorgonia"

type activation int

const (
	identity activation = iota
	sigmoid
	tanh
	relu
	leakyRelu
	elu
	softmax
	cube
)

var maxact = cube

var internalmaps = map[activation]ActivationFunction{
	identity:  nil,
	sigmoid:   G.Sigmoid,
	tanh:      G.Tanh,
	relu:      G.Rectify,
	leakyRelu: nil, // TODO
	elu:       nil, //TODO
	// softmax:   G.SoftMax, // TODO
	cube: G.Cube,
}
