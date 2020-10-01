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
	positionEmbeddings  *nn.Embedding
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

	embeddingConfig := nn.DefaultEmbeddingConfig()
	embeddingConfig.PaddingIdx = 1

	wordEmbeddings := nn.NewEmbedding(p.Sub("word_embeddings"), config.VocabSize, config.HiddenSize, embeddingConfig)
	positionEmbeddings := nn.NewEmbedding(p.Sub("position_embeddings"), config.MaxPositionEmbeddings, config.HiddenSize, nn.DefaultEmbeddingConfig())
	tokenTypeEmbeddings := nn.NewEmbedding(p.Sub("token_type_embeddings"), config.TypeVocabSize, config.HiddenSize, nn.DefaultEmbeddingConfig())

	layerNormConfig := nn.DefaultLayerNormConfig()
	layerNormConfig.Eps = 1e-12
	layerNorm := nn.NewLayerNorm(p.Sub("LayerNorm"), []int64{config.HiddenSize}, layerNormConfig)
	dropout := util.NewDropout(config.HiddenDropoutProb)

	return &RobertaEmbeddings{
		wordEmbeddings:      &wordEmbeddings,
		positionEmbeddings:  &positionEmbeddings,
		tokenTypeEmbeddings: &tokenTypeEmbeddings,
		layerNorm:           &layerNorm,
		dropout:             dropout,
	}
}

// ForwardT forwards pass through the embedding layer.
// This differs from the original BERT embeddings in how the position ids are calculated when not provided.
//
// Params:
// 	- `inputIds` - Optional input tensor of shape (batch size, sequence length).
// 		If None, pre-computed embeddings must be provided (see `inputEmbeds`)
// 	- `tokenTypeIds` -Optional segment id of shape (batch size, sequence length).
// 		Convention is value of 0 for the first sentence (incl. [SEP]) and 1 for the second sentence. If None set to 0.
// 	- `positionIds` - Optional position ids of shape (batch size, sequence length).
// 		If None, will be incremented from 0.
// 	- `inputEmbeds` - Optional pre-computed input embeddings of shape (batch size, sequence length, hidden size).
// 		If None, input ids must be provided (see `inputIds`)
// 	- `train` - boolean flag to turn on/off the dropout layers in the model.
//		Should be set to false for inference.
//
// Returns:
// 	- `embeddedOutput` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
func (re *RobertaEmbeddings) ForwardT(inputIds, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (ts.Tensor, string) {

	panic("not implemented yet")
}
