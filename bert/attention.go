package bert

import (
	"log"
	"math"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"

	"github.com/sugarme/transformer/common"
)

// BertSelfAttention:
//===================

type BertSelfAttention struct {
	NumAttentionHeads int64
	AttentionHeadSize int64
	Dropout           common.Dropout
	OutputAttentions  bool
	Query             nn.Linear
	Key               nn.Linear
	Value             nn.Linear
}

// NewBertSelfAttention creates a new `BertSelfAttention`
func NewBertSelfAttention(p nn.Path, config BertConfig) BertSelfAttention {
	if config.HiddenSize%config.NumAttentionHeads != 0 {
		log.Fatal("Hidden size is not a multiple of the number of attention heads.")
	}

	lconfig := nn.DefaultLinearConfig()
	query := nn.NewLinear(p.Sub("query"), config.HiddenSize, config.HiddenSize, lconfig)
	key := nn.NewLinear(p.Sub("key"), config.HiddenSize, config.HiddenSize, lconfig)
	value := nn.NewLinear(p.Sub("value"), config.HiddenSize, config.HiddenSize, lconfig)

	dropout := common.NewDropout(config.AttentionProbsDropoutProb)
	attentionHeadSize := int64(config.HiddenSize) / config.NumAttentionHeads
	outputAttentions := config.OutputAttentions

	return BertSelfAttention{
		NumAttentionHeads: config.NumAttentionHeads,
		AttentionHeadSize: attentionHeadSize,
		Dropout:           dropout,
		OutputAttentions:  outputAttentions,
		Query:             query,
		Key:               key,
		Value:             value,
	}

}

func (bsa *BertSelfAttention) splitHeads(x ts.Tensor, bs, dimPerHead int64) (retVal ts.Tensor) {

	xview := x.MustView([]int64{bs, -1, bsa.NumAttentionHeads, dimPerHead}, false)

	return xview.MustTranspose(1, 2, true)
}

func (bsa *BertSelfAttention) flatten(x ts.Tensor, bs, dimPerHead int64) (retVal ts.Tensor) {

	xT := x.MustTranspose(1, 2, false)
	xCon := xT.MustContiguous(true)
	retVal = xCon.MustView([]int64{bs, -1, bsa.NumAttentionHeads * dimPerHead}, true)

	return retVal
}

// ForwardT implements ModuleT interface for BertSelfAttention
//
// NOTE. mask, encoderHiddenStates, encoderMask are  optional tensors
// for `None` value, `ts.None` can be used.
func (bsa *BertSelfAttention) ForwardT(hiddenStates, mask, encoderHiddenStates, encoderMask ts.Tensor, train bool) (retVal, retValOpt ts.Tensor) {

	key := bsa.Key.Forward(hiddenStates)
	value := bsa.Value.Forward(hiddenStates)

	if encoderHiddenStates.MustDefined() {
		key = bsa.Key.Forward(encoderHiddenStates)
		value = bsa.Value.Forward(encoderHiddenStates)
	}

	bs := hiddenStates.MustSize()[0]

	hiddenStatesQ := hiddenStates.Apply(bsa.Query)
	query := bsa.splitHeads(hiddenStatesQ, bs, bsa.AttentionHeadSize)
	hiddenStatesQ.MustDrop()
	keyLayer := bsa.splitHeads(key, bs, bsa.AttentionHeadSize)
	key.MustDrop()
	valueLayer := bsa.splitHeads(value, bs, bsa.AttentionHeadSize)
	value.MustDrop()

	size := math.Sqrt(float64(bsa.AttentionHeadSize))
	queryLayer := query.MustDiv1(ts.Float64Scalar(size), true)

	// Calculate score
	var scores ts.Tensor
	if mask.MustDefined() {
		keyLayerT := keyLayer.MustTranspose(-1, -2, true)
		keyLayerT.MustAdd_(mask)
		scores = queryLayer.MustMatmul(keyLayerT, true)
	} else {
		keyLayerT := keyLayer.MustTranspose(-1, -2, true)
		scores = queryLayer.MustMatmul(keyLayerT, true)
	}

	weights := scores.MustSoftmax(-1, gotch.Float, true).ApplyT(bsa.Dropout, train)

	weightsMul := weights.MustMatmul(valueLayer, false)
	context := bsa.flatten(weightsMul, bs, bsa.AttentionHeadSize)
	weightsMul.MustDrop()

	if !bsa.OutputAttentions {
		weights.MustDrop()
		return context, ts.None
	} else {
		return context, weights
	}

}

// BertSelfOutput:
//================

type BertSelfOutput struct {
	Linear    nn.Linear
	LayerNorm nn.LayerNorm
	Dropout   common.Dropout
}

func NewBertSelfOutput(p nn.Path, config BertConfig) (retVal BertSelfOutput) {

	path := p.Sub("dense")
	lconfig := nn.DefaultLinearConfig()
	linear := nn.NewLinear(path, config.HiddenSize, config.HiddenSize, lconfig)

	layerNormConfig := nn.DefaultLayerNormConfig()
	layerNormConfig.Eps = 1e-12

	layerNorm := nn.NewLayerNorm(p.Sub("LayerNorm"), []int64{config.HiddenSize}, layerNormConfig)
	dropout := common.NewDropout(config.HiddenDropoutProb)

	return BertSelfOutput{linear, layerNorm, dropout}
}

func (bso *BertSelfOutput) ForwardT(hiddenStates ts.Tensor, inputTensor ts.Tensor, train bool) (retVal ts.Tensor) {

	state1 := inputTensor.MustAdd(hiddenStates, false)
	state2 := state1.Apply(bso.Linear)
	state3 := state2.ApplyT(bso.Dropout, train)

	retVal = state3.Apply(bso.LayerNorm)
	state1.MustDrop()
	state2.MustDrop()
	state3.MustDrop()

	return retVal
}

// BertAttention:
//===============

type BertAttention struct {
	Bsa    BertSelfAttention
	Output BertSelfOutput
}

func NewBertAttention(p nn.Path, config BertConfig) (retVal BertAttention) {
	self := NewBertSelfAttention(p.Sub("self"), config)
	output := NewBertSelfOutput(p.Sub("output"), config)

	return BertAttention{self, output}
}

func (ba BertAttention) ForwardT(hiddenStates, mask, encoderHiddenStates, encoderMask ts.Tensor, train bool) (retVal, RetValOpt ts.Tensor) {

	selfOutput, attentionWeights := ba.Bsa.ForwardT(hiddenStates, mask, encoderHiddenStates, encoderMask, train)
	selfOutput = ba.Output.ForwardT(selfOutput, hiddenStates, train)

	return selfOutput, attentionWeights
}

// BertIntermedate:
//=================

type BertIntermediate struct {
	Lin        nn.Linear
	Activation common.ActivationFn
}

func NewBertIntermediate(p nn.Path, config BertConfig) (retVal BertIntermediate) {
	lconfig := nn.DefaultLinearConfig()
	lin := nn.NewLinear(p.Sub("dense"), config.HiddenSize, config.IntermediateSize, lconfig)

	return BertIntermediate{lin, config.HiddenAct}
}

func (bi *BertIntermediate) Forward(hiddenStates ts.Tensor) (retVal ts.Tensor) {

	states := hiddenStates.Apply(bi.Lin)

	retVal = bi.Activation.Fwd(states)
	states.MustDrop()

	return retVal
}

// BertOutput:
//============

type BertOutput struct {
	Lin       nn.Linear
	LayerNorm nn.LayerNorm
	Dropout   common.Dropout
}

func NewBertOutput(p nn.Path, config BertConfig) (retVal BertOutput) {
	lconfig := nn.DefaultLinearConfig()
	lin := nn.NewLinear(p.Sub("dense"), config.IntermediateSize, config.HiddenSize, lconfig)

	layerNormConfig := nn.DefaultLayerNormConfig()
	layerNormConfig.Eps = 1e-12
	layerNorm := nn.NewLayerNorm(p.Sub("LayerNorm"), []int64{config.HiddenSize}, layerNormConfig)

	dropout := common.NewDropout(config.HiddenDropoutProb)

	return BertOutput{lin, layerNorm, dropout}
}

func (bo BertOutput) ForwardT(hiddenStates, inputTensor ts.Tensor, train bool) (retVal ts.Tensor) {

	state1 := hiddenStates.Apply(bo.Lin)
	state2 := state1.ApplyT(bo.Dropout, train)
	state3 := inputTensor.MustAdd(state3, false)

	retVal = state3.Apply(bo.LayerNorm)

	state1.MustDrop()
	state2.MustDrop()
	state3.MustDrop()

	return retVal
}
