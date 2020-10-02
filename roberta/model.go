// roberta package implements Roberta transformer model.
package roberta

import (
	// "fmt"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"

	"github.com/sugarme/transformer/bert"
	"github.com/sugarme/transformer/util"
)

// RobertaLMHead holds data of Roberta LM head.
type RobertaLMHead struct {
	dense     *nn.Linear
	decoder   *util.LinearNoBias
	layerNorm *nn.LayerNorm
	bias      ts.Tensor
}

// NewRobertaLMHead creates new RobertaLMHead.
func NewRobertaLMHead(p nn.Path, config *bert.BertConfig) *RobertaLMHead {
	dense := nn.NewLinear(p.Sub("dense"), config.HiddenSize, config.HiddenSize, nn.DefaultLinearConfig())

	layerNormConfig := nn.DefaultLayerNormConfig()
	layerNormConfig.Eps = 1e-12
	layerNorm := nn.NewLayerNorm(p.Sub("layer_norm"), []int64{config.HiddenSize}, layerNormConfig)

	decoder := util.NewLinearNoBias(p.Sub("decoder"), config.HiddenSize, config.VocabSize, util.DefaultLinearNoBiasConfig())

	bias := p.NewVar("bias", []int64{config.VocabSize}, nn.NewKaimingUniformInit())

	return &RobertaLMHead{
		dense:     &dense,
		decoder:   decoder,
		layerNorm: &layerNorm,
		bias:      bias,
	}
}

// Foward forwards pass through RobertaLMHead model.
func (rh *RobertaLMHead) Forward(hiddenStates ts.Tensor) ts.Tensor {
	gelu := util.NewGelu()
	appliedDense := hiddenStates.Apply(rh.dense)
	geluFwd := gelu.Fwd(appliedDense)
	appliedLN := geluFwd.Apply(rh.layerNorm)
	appliedDecoder := appliedLN.Apply(rh.decoder)
	appliedBias := appliedDecoder.MustAdd(rh.bias, true)

	geluFwd.MustDrop()
	appliedDense.MustDrop()
	appliedLN.MustDrop()

	return appliedBias
}

// RobertaForMaskedLM holds data for Roberta masked language model.
//
// Base RoBERTa model with a RoBERTa masked language model head to predict
// missing tokens.
type RobertaForMaskedLM struct {
	roberta *bert.BertModel
	lmHead  *RobertaLMHead
}

// NewRobertaForMaskedLM builds a new RobertaForMaskedLM.
func NewRobertaForMaskedLM(p nn.Path, config *bert.BertConfig) *RobertaForMaskedLM {
	roberta := bert.NewBertModel(p.Sub("roberta"), config)
	lmHead := NewRobertaLMHead(p.Sub("lm_head"), config)

	return &RobertaForMaskedLM{
		roberta: roberta,
		lmHead:  lmHead,
	}
}

// Forwad forwads pass through the model.
//
// Params:
// 	+ `inputIds`: Optional input tensor of shape (batch size, sequence length).
//		If None, pre-computed embeddings must be provided (see inputEmbeds).
// 	+ `mask`: Optional mask of shape (batch size, sequence length).
// 		Masked position have value 0, non-masked value 1. If None set to 1.
// 	+ `tokenTypeIds`: Optional segment id of shape (batch size, sequence length).
//		Convention is value of 0 for the first sentence (incl. </s>) and 1 for the
//		second sentence. If None set to 0.
// 	+ `positionIds`: Optional position ids of shape (batch size, sequence length).
//		If None, will be incremented from 0.
// 	+ `inputEmbeds`: Optional pre-computed input embeddings of shape (batch size,
//		sequence length, hidden size). If None, input ids must be provided (see inputIds).
// 	+ `encoderHiddenStates`: Optional encoder hidden state of shape (batch size,
// 		encoder sequence length, hidden size). If the model is defined as a decoder and
// 		the encoder hidden states is not None, used in the cross-attention layer as
// 		keys and values (query from the decoder).
// 	+ `encoderMask`: Optional encoder attention mask of shape (batch size, encoder sequence length).
// 		If the model is defined as a decoder and the *encoder_hidden_states* is not None,
// 		used to mask encoder values. Positions with value 0 will be masked.
// 	+ `train`: boolean flag to turn on/off the dropout layers in the model.
//		Should be set to false for inference.
//
// Returns:
// 	+ `output`: tensor of shape (batch size, numLabels, vocab size)
// 	+ `hiddenStates`: optional slice of tensors of length numHiddenLayers with shape
// 		(batch size, sequence length, hidden size).
// 	+ `attentions`:  optional slice of tensors of length num hidden layers with shape
// 		(batch size, sequence length, hidden size).
// 	+ `err`: error
func (mlm *RobertaForMaskedLM) Forward(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, encoderHiddenStates, encoderMask ts.Tensor, train bool) (output ts.Tensor, hiddenStates, attentions []ts.Tensor, err error) {

	hiddenState, _, allHiddenStates, allAttentions, err := mlm.roberta.ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, encoderHiddenStates, encoderMask, train)

	if err != nil {
		return ts.None, nil, nil, err
	}

	predictionScores := mlm.lmHead.Forward(hiddenState)

	return predictionScores, allHiddenStates, allAttentions, nil
}
