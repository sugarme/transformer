package bert

import (
	"fmt"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

// `BertLayer`:
//=============

// BertLayer defines a layer in BERT encoder
type BertLayer struct {
	Attention      *BertAttention
	IsDecoder      bool
	CrossAttention *BertAttention
	Intermediate   *BertIntermediate
	Output         *BertOutput
}

// NewBertLayer creates a new BertLayer.
func NewBertLayer(p nn.Path, config *BertConfig) *BertLayer {
	path := p.Sub("attention")
	attention := NewBertAttention(path, config)
	var (
		isDecoder      bool = false
		crossAttention *BertAttention
	)

	if config.IsDecoder {
		isDecoder = true
		attPath := p.Sub("cross_attention")
		crossAttention = NewBertAttention(attPath, config)
	}

	intermediatePath := p.Sub("intermediate")
	intermediate := NewBertIntermediate(intermediatePath, config)
	outputPath := p.Sub("output")
	output := NewBertOutput(outputPath, config)

	return &BertLayer{attention, isDecoder, crossAttention, intermediate, output}
}

// ForwardT forwards pass through the model.
func (bl *BertLayer) ForwardT(hiddenStates, mask, encoderHiddenStates, encoderMask ts.Tensor, train bool) (retVal, retValOpt1, retValOpt2 ts.Tensor) {
	var (
		attentionOutput       ts.Tensor
		attentionWeights      ts.Tensor
		crossAttentionWeights ts.Tensor
	)

	if bl.IsDecoder && encoderHiddenStates.MustDefined() {
		var attentionOutputTmp ts.Tensor
		attentionOutputTmp, attentionWeights = bl.Attention.ForwardT(hiddenStates, mask, ts.None, ts.None, train)
		attentionOutput, crossAttentionWeights = bl.CrossAttention.ForwardT(attentionOutputTmp, mask, encoderHiddenStates, encoderMask, train)
		attentionOutputTmp.MustDrop()
	} else {
		attentionOutput, attentionWeights = bl.Attention.ForwardT(hiddenStates, mask, ts.None, ts.None, train)
		crossAttentionWeights = ts.None
	}

	outputTmp := bl.Intermediate.Forward(attentionOutput)
	output := bl.Output.ForwardT(outputTmp, attentionOutput, train)
	attentionOutput.MustDrop()
	outputTmp.MustDrop()

	return output, attentionWeights, crossAttentionWeights
}

// `BertEncoder`:
//===============

//BertEncoder defines an encoder for BERT model
type BertEncoder struct {
	OutputAttentions   bool
	OutputHiddenStates bool
	Layers             []BertLayer
}

// NewBertEncoder creates a new BertEncoder.
func NewBertEncoder(p nn.Path, config *BertConfig) *BertEncoder {
	path := p.Sub("layer")
	outputAttentions := false
	if config.OutputAttentions {
		outputAttentions = true
	}

	outputHiddenStates := false
	if config.OutputHiddenStates {
		outputHiddenStates = true
	}

	var layers []BertLayer
	for lIdx := 0; lIdx < int(config.NumHiddenLayers); lIdx++ {
		layers = append(layers, *NewBertLayer(path.Sub(fmt.Sprintf("%v", lIdx)), config))
	}

	return &BertEncoder{outputAttentions, outputHiddenStates, layers}

}

// ForwardT forwards pass through the model.
func (be *BertEncoder) ForwardT(hiddenStates, mask, encoderHiddenStates, encoderMask ts.Tensor, train bool) (retVal ts.Tensor, retValOpt1, retValOpt2 []ts.Tensor) {
	var (
		allHiddenStates []ts.Tensor
		allAttentions   []ts.Tensor
		hiddenState     ts.Tensor
	)

	hiddenState = hiddenStates.MustDetach(false)

	for _, layer := range be.Layers {
		if allHiddenStates != nil {
			allHiddenStates = append(allHiddenStates, hiddenState)
		}

		// allHiddenStates = append(allHiddenStates, hiddenState)

		stateTmp, attnWeightsTmp, _ := layer.ForwardT(hiddenState, mask, encoderHiddenStates, encoderMask, train)
		hiddenState.MustDrop()
		hiddenState = stateTmp

		if allAttentions != nil {
			allAttentions = append(allAttentions, attnWeightsTmp)
		}
	}

	return hiddenState, allHiddenStates, allAttentions
}

// `BertPooler`:
//==============

// BertPooler defines a linear layer which can be applied to the
// first element of the sequence(`[MASK]` token)
type BertPooler struct {
	Lin *nn.Linear
}

// NewBertPooler creates a new BertPooler.
func NewBertPooler(p nn.Path, config *BertConfig) *BertPooler {
	path := p.Sub("dense")
	lconfig := nn.DefaultLinearConfig()
	lin := nn.NewLinear(path, config.HiddenSize, config.HiddenSize, lconfig)

	return &BertPooler{&lin}
}

// Forward forwards pass through the model.
func (bp *BertPooler) Forward(hiddenStates ts.Tensor) (retVal ts.Tensor) {

	selectTs := hiddenStates.MustSelect(1, 0, false)
	tmp := selectTs.Apply(bp.Lin)
	selectTs.MustDrop()
	retVal = tmp.MustTanh(true)
	return retVal
}
