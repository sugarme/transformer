package nn

import (
	"log"
	// "github.com/pkg/errors"

	"github.com/sugarme/sermo/util/ann"

	G "gorgonia.org/gorgonia"
	ts "gorgonia.org/tensor"
)

type LayerNormConfig struct {
	CudnnEnable       bool
	Eps               float64
	ElementWiseAffine bool
	WsInit            InitT
	BsInit            InitT
}

func DefaultLayerNormConfig() *LayerNormConfig {
	return &LayerNormConfig{
		CudnnEnable:       true,
		Eps:               1e-5,
		ElementWiseAffine: true,
		WsInit:            1.0,
		BsInit:            0.0,
	}
}

// LayerNorm is a layer normalization layer
type LayerNorm struct {
	Config          *LayerNormConfig
	Ws              ts.Tensor // weight
	Bs              ts.Tensor // bias
	NormalizedShape []int
}

// NewLayerNorm creates a layer normalization layer
func NewLayerNorm(path Path, normalizedShape []int, config *LayerNormConfig) *LayerNorm {
	var ws, bs ts.Tensor
	switch {
	case config.ElementWiseAffine == true:
		ws = path.Var("weight", normalizedShape, config.WsInit)
		bs = path.Var("bias", normalizedShape, config.BsInit)
	case config.ElementWiseAffine == false:
		ws = nil
	}

	return &LayerNorm{config, ws, bs, normalizedShape}

}

// TODO: use ts.Tensor??
func (ln *LayerNorm) Forward(a G.Input) G.Result {

	norm, err := ann.ConsLayerNorm(a, ann.WithName("Norm"), ann.WithEps(ln.Config.Eps))
	if err != nil {
		log.Fatal("Cannot construct layer normalization layer.")
	}

	return norm.Fwd(a)

}

/* fn forward(&self, xs: &Tensor) -> Tensor {
 *     Tensor::layer_norm(
 *         xs,
 *         self.normalized_shape.as_slice(),
 *         self.ws.as_ref(),
 *         self.bs.as_ref(),
 *         self.config.eps,
 *         self.config.cudnn_enabled,
 *     )
 * } */
