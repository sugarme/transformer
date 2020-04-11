// +build !cuda

package ann

import (
	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Conv represents a convolution layer
type Conv struct{}

// Model will return the gorgonia.Nodes associated with this convolution layer
func (l *Conv) Model() G.Nodes {
	panic("not implemented")
}

// Fwd runs the equation forwards
func (l *Conv) Fwd(x G.Input) G.Result {
	panic("not implemented")
}

// Type will return the hm.Type of the convolution layer
func (l *Conv) Type() hm.Type {
	panic("not implemented")
}

// Shape will return the tensor.Shape of the convolution layer
func (l *Conv) Shape() tensor.Shape {
	panic("not implemented")
}

// Name will return the name of the convolution layer
func (l *Conv) Name() string {
	panic("not implemented")
}

// Describe will describe a convolution layer
func (l *Conv) Describe() {
	panic("not implemented")
}
