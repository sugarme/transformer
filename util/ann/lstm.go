package ann

import (
	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// LSTM represents a static LSTM model
type LSTM struct{}

// Model will return the gorgonia.Nodes associated with this LSTM
func (l *LSTM) Model() G.Nodes {
	panic("not implemented")
}

// Fwd runs the equation forwards
func (l *LSTM) Fwd(x G.Input) G.Result {
	panic("not implemented")
}

// Type will return the hm.Type of the LSTM
func (l *LSTM) Type() hm.Type {
	panic("not implemented")
}

// Shape will return the tensor.Shape of the LSTM
func (l *LSTM) Shape() tensor.Shape {
	panic("not implemented")
}

// Name will return the name of the LSTM
func (l *LSTM) Name() string {
	panic("not implemented")
}

// Describe will describe a LSTM
func (l *LSTM) Describe() {
	panic("not implemented")
}
