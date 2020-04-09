package nn

import (
	// "github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	// "gorgonia.org/tensor"
)

// Layer represents a neural network layer.
// λ
type Layer interface {
	// σ - The weights are the "free variables" of a function
	Model() G.Nodes

	// Fwd represents the forward application of inputs
	// x.t
	Fwd(x G.Input) G.Result

	// meta stuff. This stuff is just placholder for more advanced things coming

	Term

	Type() hm.Type

	Shape() tensor.Shape

	// Serialization stuff

	// Describe returns the protobuf definition of a Layer that conforms to the ONNX standard
	Describe() // some protobuf things TODO
}

// TODO: implement it
// See https://github.com/gorgonia/golgi/blob/master/norm.go
type LayerNorm struct{}

var (
	_ Layer = (*layerNorm)(nil)
)

// layerNorm performs layer normalization as per https://arxiv.org/abs/1607.06450
type layerNorm struct {
	FC
	epsNode *G.Node
	eps     float64
}
