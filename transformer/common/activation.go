package common

import G "gorgonia.org/gorgonia"

// This code copied from "gorgonia.org/golgi"

// ActivationFunction represents an activation function
// Note: This may become an interface once we've worked through all the linter errors
type ActivationFunction func(*G.Node) (*G.Node, error)

type Activation int

const (
	identity Activation = iota
	sigmoid
	tanh
	relu
	leakyRelu
	elu
	softmax
	cube
)

var maxact = cube

var internalmaps = map[Activation]ActivationFunction{
	identity:  nil,
	sigmoid:   G.Sigmoid,
	tanh:      G.Tanh,
	relu:      G.Rectify,
	leakyRelu: nil, // TODO
	elu:       nil, //TODO
	// softmax:   G.SoftMax, // TODO
	cube: G.Cube,
}
