package bert

import (
	"log"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"

	"github.com/sugarme/gotch/nn"
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

func (bl *BertLayer) ForwardT(hiddenState *G.Node, mask, encoderHiddenStates, encoderMask *G.Node, train bool) (*G.Node, *G.Node, *G.Node, error) {
	var (
		attentionOuput        *G.Node
		attentionWeights      *G.Node
		crossAttentionWeights *G.Node
		err                   error
	)

	attentionOuput, attentionWeights, err = bl.Attention.ForwardT(hiddenState, mask, nil, nil, train)
	if err != nil {
		return nil, nil, nil, err
	}

	if bl.IsDecoder && encoderHiddenStates != nil {
		attentionOuput, attentionWeights, err = bl.Attention.ForwardT(hiddenState, mask, nil, nil, train)
		attentionOuput, crossAttentionWeights, err = bl.CrossAttention.ForwardT(attentionOuput, mask, encoderHiddenStates, encoderMask, train)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	output := bl.Intermediate.Forward(attentionOuput)

	return output, attentionWeights, crossAttentionWeights, nil
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
func (be *BertEncoder) ForwardT(hiddenState, mask, encoderHiddenStates, encoderMask *G.Node, train bool) (*G.Node, []*G.Node, []*G.Node, error) {
	var (
		attentionWeights               *G.Node
		allHiddenStates, allAttentions []*G.Node
		err                            error
	)

	if be.OutputHiddenStates {
		allHiddenStates = make([]*G.Node, 0) // initialize it
	}
	if be.OutputAttentions {
		allAttentions = make([]*G.Node, 0)
	}

	for _, layer := range be.Layers {
		if allHiddenStates != nil {
			allHiddenStates = append(allHiddenStates, hiddenState)
		}

		hiddenState, attentionWeights, _, err = layer.ForwardT(hiddenState, mask, encoderHiddenStates, encoderMask, train)
		if err != nil {
			return nil, nil, nil, err
		}

		if allAttentions != nil {
			allAttentions = append(allAttentions, attentionWeights)
		}
	}

	return hiddenState, allHiddenStates, allAttentions, nil
}

// `BertPooler`:
//==============

// BertPooler defines a linear layer which can be applied to the
// first element of the sequence(`[MASK]` token)
type BertPooler struct {
	Lin *nn.Linear
}

func NewBertPooler(p nn.Path, config *BertConfig) *BertPooler {
	path := p.Sub("dense")
	lconfig := nn.DefaultLinearConfig()
	lin := nn.NewLinear(path, config.HiddenSize, config.HiddenSize, lconfig)

	return &BertPooler{lin}
}

func (bp *BertPooler) Forward(hiddenStates *G.Node) *G.Node {
	// select row 0 (hidden_states.select(1,0))
	var slice ts.Slice
	slice = G.S(1)
	n, err := G.Slice(hiddenStates, slice)
	if err != nil {
		log.Fatal(err)
	}

	hiddenState := bp.Lin.Forward(n)
	hiddenState, err = G.Tanh(hiddenState)

	return hiddenState
}
