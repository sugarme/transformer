package nn

import (
	"log"
	"math"

	G "gorgonia.org/gorgonia"
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
	Ws *G.Node // weights
	Bs *G.Node // Bias
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

func (l *Linear) Forward(xs *G.Node) *G.Node {
	mulRes, err := G.Mul(xs, l.Ws)
	if err != nil {
		log.Fatal(err)
	}

	res, err := G.Add(mulRes, l.Bs)
	if err != nil {
		log.Fatal(err)
	}

	return res
}
