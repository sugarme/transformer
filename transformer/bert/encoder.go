package bert

import (
	// G "gorgonia.org/gorgonia"

	"github.com/sugarme/sermo/util/nn"
)

//BertEncoder defines an encoder for BERT model
type BertEncoder struct {
	OutputAttentions   bool
	OutputHiddenStates bool
	Layers             []BertLayer
}

// BertLayer defines a layer in BERT encoder
type BertLayer struct {
	Attention      BertAttention
	IsDecoder      bool
	CrossAttention BertAttention
	Intermediate   BertIntermediate
	Output         BertOutput
}

// BertPooler defines a linear layer which can be applied to the
// first element of the sequence(`[MASK]` token)
type BertPooler struct {
	Lin nn.Linear
}
