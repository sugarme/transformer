package bert

import (
	"github.com/sugarme/sermo/transformer/common"
	"github.com/sugarme/sermo/util/nn"

	// G "gorgonia.org/gorgonia"
	ts "gorgonia.org/tensor"
)

// BertEmbedding defines interface for BertModel or RoBertaModel
type BertEmbedding interface {
	ForwardT(inputIds, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (ts.Tensor, string)
}

// Dropout is a func type with signature of Gorgonia Dropout func
// type Dropout func(x *G.Node, prob float64) (retVal *G.Node, err error)

type BertEmbeddings struct {
	WordEmbeddings      *nn.Embedding
	PositionEmbeddings  *nn.Embedding
	TokenTypeEmbeddings *nn.Embedding
	LayerNorm           *nn.LayerNorm
	Dropout             *common.Dropout
}

// NewBertEmbedding builds a new BertEmbedding
// * `p` - Variable store path for the root of the BertEmbeddings model
// * `config` - `BertConfig` object defining the model architecture and vocab/hidden size
func NewBertEmbedding(p nn.Path, config BertConfig) (*BertEmbeddings, error) {
	embeddingConfig := nn.DefaultEmbeddingConfig()
	embeddingConfig.PaddingIdx = 0

	wEmbedPath := p.Sub("wordEmbeddings")
	wordEmbeddings := nn.Embed(wEmbedPath, int(config.VocabSize), int(config.HiddenSize), embeddingConfig)

	posEmbedPath := p.Sub("PositionEmbeddings")
	positionEmbeddings := nn.Embed(posEmbedPath, int(config.VocabSize), int(config.HiddenSize), embeddingConfig)

	ttEmbedPath := p.Sub("tokenTypeEmbeddings")
	tokenTypeEmbeddings := nn.Embed(ttEmbedPath, int(config.VocabSize), int(config.HiddenSize), embeddingConfig)

	layerNormConfig := nn.DefaultLayerNormConfig()
	layerNormConfig.Eps = 1e-12

	lnPath := p.Sub("LayerNorm")
	layerNorm := nn.NewLayerNorm(lnPath, []int{config.HiddenSize}, layerNormConfig)

	dropout := common.NewDropout(config.HiddenDropoutProb)
	bertEmbedding := &BertEmbeddings{wordEmbeddings, positionEmbeddings, tokenTypeEmbeddings, layerNorm, dropout}

	return bertEmbedding, nil
}

// Implement BertEmbedding interface
func (be *BertEmbeddings) ForwardT(inputIds, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (ts.Tensor, string) {

	return ts.New(), ""
}
