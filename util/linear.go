package util

import (
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

type LinearNoBiasConfig struct {
	WsInit nn.Init // interface
}

func DefaultLinearNoBiasConfig() *LinearNoBiasConfig {

	init := nn.NewKaimingUniformInit()

	return &LinearNoBiasConfig{WsInit: init}
}

type LinearNoBias struct {
	Ws *ts.Tensor
}

func NewLinearNoBias(vs *nn.Path, inDim, outDim int64, config *LinearNoBiasConfig) (*LinearNoBias, error) {

	ws, err := vs.NewVar("weight", []int64{outDim, inDim}, config.WsInit)
	if err != nil {
		return nil, err
	}

	return &LinearNoBias{
		Ws: ws,
	}, nil
}

// Forward implements Module interface for LinearNoBias
func (lnb *LinearNoBias) Forward(xs *ts.Tensor) (retVal *ts.Tensor) {
	wsT := lnb.Ws.MustT(false)
	retVal = xs.MustMatmul(wsT, false)
	wsT.MustDrop()

	return retVal
}
