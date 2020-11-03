package bert

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"

	"github.com/sugarme/transformer/util"
)

// BertEmbedding defines interface for BertModel or RoBertaModel.
type BertEmbedding interface {
	ForwardT(inputIds, tokenTypeIds, positionIds, inputEmbeds *ts.Tensor, train bool) (*ts.Tensor, error)
}

type BertEmbeddings struct {
	WordEmbeddings      *nn.Embedding
	PositionEmbeddings  *nn.Embedding
	TokenTypeEmbeddings *nn.Embedding
	LayerNorm           *nn.LayerNorm
	Dropout             *util.Dropout
}

// NewBertEmbeddings builds a new BertEmbeddings
func NewBertEmbeddings(p *nn.Path, config *BertConfig) *BertEmbeddings {
	embeddingConfig := nn.DefaultEmbeddingConfig()
	embeddingConfig.PaddingIdx = 0

	wEmbedPath := p.Sub("word_embeddings")
	wordEmbeddings := nn.NewEmbedding(wEmbedPath, config.VocabSize, config.HiddenSize, embeddingConfig)

	posEmbedPath := p.Sub("position_embeddings")
	positionEmbeddings := nn.NewEmbedding(posEmbedPath, config.MaxPositionEmbeddings, config.HiddenSize, embeddingConfig)

	ttEmbedPath := p.Sub("token_type_embeddings")
	tokenTypeEmbeddings := nn.NewEmbedding(ttEmbedPath, config.TypeVocabSize, config.HiddenSize, embeddingConfig)

	layerNormConfig := nn.DefaultLayerNormConfig()
	layerNormConfig.Eps = 1e-12

	lnPath := p.Sub("LayerNorm")
	layerNorm := nn.NewLayerNorm(lnPath, []int64{config.HiddenSize}, layerNormConfig)

	dropout := util.NewDropout(config.HiddenDropoutProb)

	return &BertEmbeddings{wordEmbeddings, positionEmbeddings, tokenTypeEmbeddings, layerNorm, dropout}
}

// ForwardT implements BertEmbedding interface, passes through the embedding layer
func (be *BertEmbeddings) ForwardT(inputIds, tokenTypeIds, positionIds, inputEmbeds *ts.Tensor, train bool) (*ts.Tensor, error) {

	var (
		inputEmbeddings       *ts.Tensor
		inputShape            []int64
		deleteInputEmbeddings bool
	)

	if inputIds.MustDefined() {
		if inputEmbeds.MustDefined() {
			err := fmt.Errorf("Only one of input Ids or input embeddings may be set.")
			return nil, err
		} else {
			inputEmbeddings = inputIds.ApplyT(be.WordEmbeddings, train)
			// mark to delete later
			deleteInputEmbeddings = true
			inputShape = inputIds.MustSize()
		}
	} else {
		if inputEmbeds.MustDefined() {
			inputEmbeddings = inputEmbeds
			size := inputEmbeds.MustSize()
			inputShape = []int64{size[0], size[1]}
		} else {
			err := fmt.Errorf("Only one of input Ids or input embeddings may be set.")
			return nil, err
		}
	}

	seqLength := inputEmbeddings.MustSize()[1]

	var posIds *ts.Tensor
	if positionIds.MustDefined() {
		posIds = positionIds
	} else {
		seqLenTs := ts.IntScalar(seqLength)
		tmp1 := ts.MustArange(seqLenTs, gotch.Int64, inputEmbeddings.MustDevice())
		seqLenTs.MustDrop()
		tmp2 := tmp1.MustUnsqueeze(0, true)
		posIds = tmp2.MustExpand(inputShape, true, true)
	}

	var tokTypeIds *ts.Tensor
	if tokenTypeIds.MustDefined() {
		tokTypeIds = tokenTypeIds
	} else {
		tokTypeIds = ts.MustZeros(inputShape, gotch.Int64, inputEmbeddings.MustDevice())
	}

	posEmbeddings := posIds.Apply(be.PositionEmbeddings)
	// delete if didn't come from input param
	if !positionIds.MustDefined() {
		posIds.MustDrop()
	}

	tokEmbeddings := tokTypeIds.Apply(be.TokenTypeEmbeddings)
	// delete if didn't come from input param
	if !tokenTypeIds.MustDefined() {
		tokTypeIds.MustDrop()
	}

	input := inputEmbeddings.MustAdd(posEmbeddings, false)
	// delete if didn't come from input param
	if deleteInputEmbeddings {
		inputEmbeddings.MustDrop()
	}

	posEmbeddings.MustDrop()
	addedInput := input.MustAdd(tokEmbeddings, true)
	tokEmbeddings.MustDrop()

	retTmp1 := addedInput.Apply(be.LayerNorm)
	addedInput.MustDrop()
	retVal := retTmp1.ApplyT(be.Dropout, train)
	retTmp1.MustDrop()

	return retVal, nil
}
