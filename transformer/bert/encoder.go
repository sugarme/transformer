package bert

import (
	"reflect"
	"runtime"

	"github.com/sugarme/sermo/util/nn"

	ts "gorgonia.org/tensor"
	// G "gorgonia.org/gorgonia"
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

func NewBertLayer(p nn.Path, config *BertConfig) *BertLayer {
	path := p.Sub("attention")
	attention := NewBertAttention(path, config)
	var (
		isDecoder      bool = false
		crossAttention *BertAttention
	)

	if config.IsDecoder {
		isDecoder = true
		attPath := path.Sub("crossAttention")
		crossAttention = NewBertAttention(attPath, config)
	}

	intermediatePath := path.Sub("intermediate")
	intermediate := NewBertIntermediate(intermediatePath, config)
	outputPath := path.Sub("output")
	output := NewBertOutput(outputPath, config)

	return &BertLayer{attention, isDecoder, crossAttention, intermediate, output}
}

func (bl *BertLayer) ForwardT(hiddenState *ts.Tensor, train bool, opts ...TensorOpt) (*ts.Tensor, []*ts.Tensor, []*ts.Tensor) {
	var (
		mask, encoderHiddenStates, encoderMask *ts.Tensor
		attentionOuput                         *ts.Tensor
		attentionWeights                       []*ts.Tensor
		crossAttentionWeights                  []*ts.Tensor
	)

	for _, o := range opts {
		switch {
		case runtime.FuncForPC(reflect.ValueOf(o).Pointer()).Name() == "MaskOpt":
			mask = o()
		case runtime.FuncForPC(reflect.ValueOf(o).Pointer()).Name() == "EncoderHiddenStateOpt":
			encoderHiddenStates = o()
		case runtime.FuncForPC(reflect.ValueOf(o).Pointer()).Name() == "EncoderMaskOpt":
			encoderMask = o()
		}
	}

	maskOpt := MaskTensorOpt(mask)

	if bl.IsDecoder && encoderHiddenStates != nil {
		attentionOuput, attentionWeights = bl.Attention.ForwardT(hiddenState, train)

		encoderHiddenStatesOpt := EncoderHiddenStateTensorOpt(encoderHiddenStates)
		encoderMaskOpt := MaskTensorOpt(encoderMask)

		attentionOuput, attentionWeights = bl.Attention.ForwardT(attentionOuput, train, maskOpt)
		attentionOuput, crossAttentionWeights = bl.CrossAttention.ForwardT(attentionOuput, train, maskOpt, encoderHiddenStatesOpt, encoderMaskOpt)

	} else {
		attentionOuput, attentionWeights = bl.Attention.ForwardT(hiddenState, train, maskOpt)
	}

	output := bl.Intermediate.ForwardT(attentionOuput)

	return output, attentionWeights, crossAttentionWeights
}

// `BertEncoder`:
//===============

//BertEncoder defines an encoder for BERT model
type BertEncoder struct {
	OutputAttentions   bool
	OutputHiddenStates bool
	Layers             []*BertLayer
}

func NewBertEncoder(p nn.Path, config *BertConfig) *BertEncoder {
	path := p.Sub("layer")
	outputAttentions := false
	if config.OutputAttentions {
		outputAttentions = true
	}

	outputHiddenStates := false
	if config.OutputAttentions {
		outputAttentions = true
	}

	var layers []*BertLayer
	for lIdx := 0; lIdx < int(config.NumHiddenLayers); lIdx++ {
		layers = append(layers, NewBertLayer(path.Sub(string(lIdx)), config))
	}

	return &BertEncoder{outputAttentions, outputHiddenStates, layers}

}

// Forward ...
func (be *BertEncoder) Forward(hiddenState *ts.Tensor, train bool, opts ...TensorOpt) (*ts.Tensor, []*ts.Tensor, []*ts.Tensor) {
	var (
		mask, encoderHiddenStates, encoderMask *ts.Tensor
		attentionWeights                       *ts.Tensor
		allHiddenStates, allAttentions         []*ts.Tensor
	)

	for _, o := range opts {
		switch {
		case runtime.FuncForPC(reflect.ValueOf(o).Pointer()).Name() == "MaskTensorOpt":
			mask = o()
		case runtime.FuncForPC(reflect.ValueOf(o).Pointer()).Name() == "EncoderHiddenStateTensorOpt":
			encoderHiddenStates = o()
		case runtime.FuncForPC(reflect.ValueOf(o).Pointer()).Name() == "EncoderMaskTensorOpt":
			encoderMask = o()
		}
	}

	if be.OutputHiddenStates {
		allHiddenStates = make([]*ts.Tensor, 0) // initialize it
	}
	if be.OutputAttentions {
		allAttentions = make([]*ts.Tensor, 0)
	}

	for _, layer := range be.Layers {
		if allHiddenStates != nil {
			allHiddenStates = append(allHiddenStates, hiddenState)
		}

		maskOpt := MaskTensorOpt(mask)
		encoderHiddenStatesOpt := EncoderHiddenStateTensorOpt(encoderHiddenStates)
		encoderMaskOpt := EncoderMaskTensorOpt(encoderMask)

		temp := layer.ForwardT(hiddenState, train, maskOpt, encoderHiddenStatesOpt, encoderMaskOpt)
		hiddenState = temp[0]
		attentionWeights = temp[1]
		if allAttentions != nil {
			allAttentions = append(allAttentions, attentionWeights)
		}
	}

	return hiddenState, allHiddenStates, allAttentions
}

// `BertPooler`:
//==============

// BertPooler defines a linear layer which can be applied to the
// first element of the sequence(`[MASK]` token)
type BertPooler struct {
	Lin nn.Linear
}

func NewBertPool(p nn.Path, config *BertConfig) *BertPooler {

	path := p.Sub("dense")
	lin := nn.NewLinear(path, config.HiddenSize, config.HiddenSize)

	return &BertPooler{lin}
}

func (bp *BertPooler) Forward(hiddenStatus *ts.Tensor) *ts.Tensor {

	// TODO: implement it
	var res ts.Tensor

	return &res

}

/*     pub fn new(p: &nn::Path, config: &BertConfig) -> BertPooler {
 *         let lin = nn::linear(&(p / "dense"), config.hidden_size, config.hidden_size, Default::default());
 *         BertPooler { lin }
 *     }
 *
 *     pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
 *         hidden_states
 *             .select(1, 0)
 *             .apply(&self.lin)
 *             .tanh()
 *     } */
