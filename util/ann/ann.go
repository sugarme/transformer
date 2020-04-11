package ann

import (
	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ByNamer is any type that allows a name to be found and returned.
//
// If a name is not found, `nil` is to be returned
type ByNamer interface {
	ByName(name string) Term
}

// Grapher is any type that can return the underlying computational graph
type Grapher interface {
	Graph() *G.ExprGraph
}

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

// Redefine redefines a layer with the given construction options. This is useful for re-initializing layers
func Redefine(l Layer, opts ...ConsOpt) (retVal Layer, err error) {
	for _, opt := range opts {
		if l, err := opt(l); err != nil {
			return l, err
		}
	}
	return l, nil
}

// Apply will apply two terms and return the resulting term
// Note: This has not yet been implemented, please do not use!
func Apply(a, b Term) (Term, error) {
	panic("STUBBED")
}
