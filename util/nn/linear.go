package nn

import (
	"log"
	"math"

	ts "gorgonia.org/tensor"
)

type LinearConfig struct {
	WsInit InitT
	BsInit InitT // optional
	Bias   bool
}

func DefaultLinearConfig() *LinearConfig {
	return &LinearConfig{
		WsInit: nil, // Should be KaimingUniform init with dims
		BsInit: nil,
		Bias:   true,
	}
}

// Linear is a fully connected layer
type Linear struct {
	Ws *ts.Tensor // weights
	Bs *ts.Tensor // Bias
}

func NewLinear(vs Path, inDim, outDim int, config *LinearConfig) *Linear {

	bound := 1.0 / (math.Sqrt(float64(inDim)))
	bsInit := NewUniform(-bound, bound)

	if config.BsInit != nil {
		bsInit = config.BsInit.([]float64)
	}

	ws := vs.Var("weight", []int{outDim, inDim}, config.WsInit)
	bs := vs.Var("bias", []int{outDim}, bsInit)

	return &Linear{
		Ws: &ws,
		Bs: &bs,
	}
}

func (l *Linear) Forward(xs *ts.Tensor) *ts.Tensor {
	mulRes, err := ts.MatMul(*xs, *l.Ws)
	if err != nil {
		log.Fatal(err)
	}

	res, err := ts.Add(mulRes, l.Bs)
	if err != nil {
		log.Fatal(err)
	}

	return &res
}
