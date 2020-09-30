package transformer

import (
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

type LinearNoBiasConfig struct {
	WsInit nn.Init
}

func DefaultLinearNoBiasConfig() (retVal LinearNoBiasConfig) {

	init := nn.NewKaimingUniformInit()

	return LinearNoBiasConfig{WsInit: init}
}

type LinearNoBias struct {
	Ws ts.Tensor
}

func NewLinearNoBias(vs nn.Path, inDim, outDim int64, config LinearNoBiasConfig) (retVal LinearNoBias) {

	return LinearNoBias{
		Ws: vs.NewVar("weight", []int64{outDim, inDim}, config.WsInit),
	}
}

// Forward implements Module interface for LinearNoBias
func (lnb LinearNoBias) Forward(xs ts.Tensor) (retVal ts.Tensor) {
	wsT := lnb.Ws.MustT(false)
	retVal = xs.MustMatmul(wsT, false)
	wsT.MustDrop()

	return retVal
}
