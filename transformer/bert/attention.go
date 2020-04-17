package bert

import (
	"log"

	"github.com/sugarme/sermo/transformer/common"
	"github.com/sugarme/sermo/util/nn"

	G "gorgonia.org/gorgonia"
)

// BertSelfAttention:
//===================

type BertSelfAttention struct {
	NumAttentionHeads int64
	AttentionHeadSize int64
	Dropout           *common.Dropout
	OutputAttentions  bool
	Query             *nn.Linear
	Key               *nn.Linear
	Value             *nn.Linear
}

// NewBertSelfAttention creates a new `BertSelfAttention`
func NewBertSelfAttention(p nn.Path, config *BertConfig) *BertSelfAttention {
	if config.HiddenSize%int(config.NumAttentionHeads) != 0 {
		log.Fatal("Hidden size is not a multiple of the number of attention heads.")
	}

	lconfig := nn.DefaultLinearConfig()
	query := nn.NewLinear(p.Sub("query"), config.HiddenSize, config.HiddenSize, lconfig)
	key := nn.NewLinear(p.Sub("key"), config.HiddenSize, config.HiddenSize, lconfig)
	value := nn.NewLinear(p.Sub("value"), config.HiddenSize, config.HiddenSize, lconfig)

	dropout := common.NewDropout(config.AttentionProbsDropoutProb)
	attentionHeadSize := int64(config.HiddenSize) / config.NumAttentionHeads
	outputAttentions := config.OutputAttentions

	return &BertSelfAttention{
		NumAttentionHeads: config.NumAttentionHeads,
		AttentionHeadSize: attentionHeadSize,
		Dropout:           dropout,
		OutputAttentions:  outputAttentions,
		Query:             query,
		Key:               key,
		Value:             value,
	}

}

func (bsa *BertSelfAttention) splitHeads(x *G.Node, bs, dimPerHead int) (retVal *G.Node, err error) {

	// 1. Reshape node
	shape := []int{bs, -1, int(bsa.NumAttentionHeads), dimPerHead}
	n, err := G.Reshape(x, shape)
	if err != nil {
		return nil, err
	}

	// 2. Transpose
	retVal, err = G.Transpose(n, 1, 2)
	if err != nil {
		return nil, err
	}

	return retVal, err
}

func (bsa *BertSelfAttention) flatten(x *G.Node, bs, dimPerHead int) (retVal *G.Node, err error) {

	shape := []int{bs, -1, int(bsa.NumAttentionHeads), dimPerHead}
	// 1. Transpose
	n, err := G.Transpose(x, 1, 2)
	if err != nil {
		return nil, err
	}

	// 2. Reshape node
	retVal, err = G.Reshape(n, shape)
	if err != nil {
		return nil, err
	}

	return retVal, err
}

// func (bsa *BertSelfAttention) ForwardT(hiddenStates *G.Node, train bool, opts ...TensorOpt) (retVal, retValOpt *G.Node, err error) {
func (bsa *BertSelfAttention) ForwardT(hiddenStates, mask, encoderHiddenStates, encoderMask *G.Node, train bool) (retVal, retValOpt *G.Node, err error) {

	keyLayer := bsa.Key.Forward(hiddenStates)
	valueLayer := bsa.Value.Forward(hiddenStates)

	if encoderHiddenStates != nil {
		// use it
		keyLayer = bsa.Key.Forward(encoderHiddenStates)
		valueLayer = bsa.Value.Forward(encoderHiddenStates)
	}

	bs := hiddenStates.DataSize()

	queryLayer, err := bsa.splitHeads(bsa.Query.Forward(hiddenStates), bs, int(bsa.AttentionHeadSize))
	if err != nil {
		return nil, nil, err
	}

	keyLayer, err = bsa.splitHeads(keyLayer, bs, int(bsa.AttentionHeadSize))
	if err != nil {
		return nil, nil, err
	}
	valueLayer, err = bsa.splitHeads(valueLayer, bs, int(bsa.AttentionHeadSize))
	if err != nil {
		return nil, nil, err
	}
	// TODO: do `queryLayer` divided by math.sqrt(bsa.AttentionHeadSize)

	// Calculate score
	keyLayerT, err := G.Transpose(keyLayer, -1, -2)
	if err != nil {
		return nil, nil, err
	}

	scores := G.Must(G.Add(G.Must(G.Mul(queryLayer, keyLayerT)), mask))

	weights, err := G.SoftMax(scores, -1)
	if err != nil {
		return nil, nil, err
	}

	context, err := bsa.flatten(G.Must(G.Mul(weights, valueLayer)), bs, int(bsa.AttentionHeadSize))
	if err != nil {
		return nil, nil, err
	}

	if !bsa.OutputAttentions {
		return context, nil, nil
	}

	return context, weights, nil

}

// BertSelfOutput:
//================

type BertSelfOutput struct {
	Linear    *nn.Linear
	LayerNorm *nn.LayerNorm
	Droput    *common.Dropout
}

func NewBertSelfOutput(p nn.Path, config *BertConfig) *BertSelfOutput {

	path := p.Sub("dense")
	lconfig := nn.DefaultLinearConfig()
	linear := nn.NewLinear(path, config.HiddenSize, config.HiddenSize, lconfig)

	layerNormConfig := nn.DefaultLayerNormConfig()
	layerNormConfig.Eps = 1e-12

	layerNorm := nn.NewLayerNorm(p.Sub("LayerNorm"), []int{config.HiddenSize}, layerNormConfig)
	dropout := NewDropout(config.HiddenDropoutProb)

	return &BertSelfOutput{linear, layerNorm, dropout}
}

func (bso *BertSelfOutput) ForwardT(hiddenStates *G.Node, inputTensor *G.Node, train bool) *G.Node {

	hiddenState := bso.Linear.Forward(hiddenStates)
	hiddenState = bso.Droput.ForwardT(hiddenStates, train)

	return hiddenState
}

// BertAttention:
//===============

type BertAttention struct {
	self   *BertSelfAttention
	output *BertSelfOutput
}

func NewBertAttention(p nn.Path, config *BertConfig) *BertAttention {
	self := NewBertSelfAttention(p.Sub("self"), config)
	output := NewBertSelfOutput(p.Sub("output"), config)

	return &BertAttention{self, output}
}

func (ba *BertAttention) ForwardT(hiddenStates *G.Node, mask, encoderHiddenStates, encoderMask *G.Node, train bool) (retVal, RetValOpt *G.Node, err error) {
	selfOutput, attentionWeights, err := ba.self.ForwardT(hiddenStates, mask, encoderHiddenStates, encoderMask, train)
	if err != nil {
		return nil, nil, err
	}

	selfOutput = ba.output.ForwardT(selfOutput, hiddenStates, train)

	return selfOutput, attentionWeights, nil
}

// BertIntermedate:
//=================

type BertIntermediate struct {
	lin        *nn.Linear
	activation common.ActivationFn
}

func NewBertIntermediate(p nn.Path, config *BertConfig) *BertIntermediate {
	lconfig := nn.DefaultLinearConfig()
	lin := nn.NewLinear(p.Sub("dense"), config.HiddenSize, int(config.IntermediateSize), lconfig)

	return &BertIntermediate{lin, config.HiddenAct}
}

// BertOutput:
//============

type BertOutput struct {
	lin       *nn.Linear
	layerNorm *nn.LayerNorm
	dropout   *common.Dropout
}

func NewBertOutput(p nn.Path, config *BertConfig) *BertOutput {
	lconfig := nn.DefaultLinearConfig()
	lin := nn.NewLinear(p.Sub("dense"), int(config.IntermediateSize), config.HiddenSize, lconfig)

	layerNormConfig := nn.DefaultLayerNormConfig()
	layerNormConfig.Eps = 1e-12
	layerNorm := nn.NewLayerNorm(p.Sub("LayerNorm"), []int{config.HiddenSize}, layerNormConfig)

	dropout := common.NewDropout(config.HiddenDropoutProb)

	return &BertOutput{lin, layerNorm, dropout}
}

func (bo *BertOutput) ForwardT(hiddenStates, inputTensor *G.Node, train bool) *G.Node {

	hiddenState := bo.lin.Forward(hiddenStates)
	hiddenState = bo.dropout.ForwardT(hiddenState, train)

	hiddenState = G.Must(G.Add(inputTensor, hiddenState))

	hiddenState = bo.layerNorm.Forward(hiddenState)

	return hiddenState
}
