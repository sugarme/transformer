package nn

import (
	"math/rand"

	ts "gorgonia.org/tensor"
)

// Configuration option for an embedding layer
type EmbeddingConfig struct {
	Sparse          bool
	ScaleGradByFreq bool
	WsInit          interface{} // can be const float64, Uniform {lo: float64, up: float64}, or KaimingUniform
	PaddingIdx      int64
}

func DefaultEmbeddingConfig() *EmbeddingConfig {

	return &EmbeddingConfig{
		Sparse:          false,
		ScaleGradByFreq: false,
		WsInit:          rand.NormFloat64(),
		PaddingIdx:      -1,
	}
}

// An embedding layer.
//
// An embedding layer acts as a simple lookup table that stores embeddings.
// This is commonly used to store word embeddings.
type Embedding struct {
	Ws     ts.Tensor
	Config EmbeddingConfig
}

func (e *Embedding) Forward(xs *ts.Tensor) *ts.Tensor {
	return &Tensor.Embedding{
		e.Ws,
		xs,
		e.Config.PaddingIdx,
		e.Config.ScaleGradByFreq,
		e.Config.Sparse,
	}
}

func Embed(vs interface{}, numEmbeddings, embeddingDim int64, config EmbeddingConfig) *Embedding {
	return &Embedding{
		Ws:     vs.Var("weight", []int64{numEmbeddings, embeddingDim}, config.WsInit),
		Config: config,
	}
}
