package bert

import (
	// "github.com/sugarme/sermo/tokenizer"
	// data "github.com/sugarme/sermo/util/data"
	"github.com/sugarme/sermo/util/nn"

	G "gorgonia.org/gorgonia"
	ts "gorgonia.org/tensor"
)

// BertEmbedding defines interface for BertModel or RoBertaModel
type BertEmbedding interface {
	ForwardT(inputIds, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (ts.Tensor, string)
}

// Dropout is a func type with signature of Gorgonia Dropout func
type Dropout func(x *G.Node, prob float64) (retVal *G.Node, err error)

type BertEmbeddings struct {
	WordEmbeddings      nn.Embedding
	PositionEmbeddings  nn.Embedding
	TokenTypeEmbeddings nn.Embedding
	LayerNorm           nn.LayerNorm
	Dropout             Dropout
}

func NewBertEmbedding(p nn.Path, config BertConfig) (BertEmbeddings, error) {

	return BertEmbeddings{}, nil
}

// Implement BertEmbedding interface
func (be *BertEmbeddings) ForwardT(inputIds, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (ts.Tensor, string) {

	return ts.Tensor{}, nil
}
