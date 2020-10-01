package roberta

import (
	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"

	"github.com/sugarme/transformer/bert"
	"github.com/sugarme/transformer/util"
)

// RobertaEmbeddings holds embedding struct for Roberta model.
// It also implements `BertEmbedding` interface for Roberta models.
type RobertaEmbeddings struct {
	wordEmbeddings      *nn.Embedding
	tokenTypeEmbeddings *nn.Embedding
	layerNorm           *nn.LayerNorm
	dropout             *util.Dropout
	paddingIndex        int64
}

func (re *RobertaEmbeddings) createPositionIdsFromInputIds(x ts.Tensor) ts.Tensor {
	mask := x.MustNe(ts.IntScalar(re.paddingIndex), false).MustTotype(gotch.Int64, true)
	cumSum := mask.MustCumsum(1, gotch.Int64, false)
	mul := cumSum.MustMul(mask, true)
	retVal := mul.MustAdd1(ts.IntScalar(re.paddingIndex), false)
	mul.MustDrop()

	return retVal
}

func (re *RobertaEmbeddings) createPositionIdsFromEmbeddings(x ts.Tensor) ts.Tensor {
	shape := x.MustSize()
	var inputShape []int64 = []int64{shape[0], shape[1]}

	positionIds := ts.MustArange1(ts.IntScalar(re.paddingIndex+1), ts.IntScalar(inputShape[0]), gotch.Int64, x.MustDevice())
	retVal := positionIds.MustUnsqueeze(0, false).MustExpand(inputShape, true, true)

	return retVal
}

// NewRobertaEmbeddings creates a new RobertaEmbeddings.
//
// Params:
// 	- `p` - Variable store path for the root of the BertEmbeddings model
// 	- `config` - `BertConfig` object defining the model architecture and vocab/hidden size.
func NewRobertaEmbeddings(p nn.Path, config *bert.BertConfig) *RobertaEmbeddings {

	// TODO: implement it

	return &RobertaEmbeddings{}
}
