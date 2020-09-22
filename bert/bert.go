package bert

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/transformer/common"
)

// BertConfig defines the BERT model architecture (i.e., number of layers,
// hidden layer size, label mapping...)
type BertConfig struct {
	HiddenAct                 string           `json:"hidden_act"`
	AttentionProbsDropoutProb float64          `json:"attention_probs_dropout_prob"`
	HiddenDropoutProb         float64          `json:"hidden_dropout_prob"`
	HiddenSize                int64            `json:"hidden_size"`
	InitializerRange          float32          `json:"initializer_range"`
	IntermediateSize          int64            `json:"intermediate_size"`
	MaxPositionEmbeddings     int64            `json:"max_position_embeddings"`
	NumAttentionHeads         int64            `json:"num_attention_heads"`
	NumHiddenLayers           int64            `json:"num_hidden_layers"`
	TypeVocabSize             int64            `json:"type_vocab_size"`
	VocabSize                 int64            `json:"vocab_size"`
	OutputAttentions          bool             `json:"output_attentions"`
	OutputHiddenStates        bool             `json:"output_hidden_states"`
	IsDecoder                 bool             `json:"is_decoder"`
	Id2Label                  map[int64]string `json:"id_2_label"`
	Label2Id                  map[string]int64 `json:"label_2_id"`
	NumLabels                 int64            `json:"num_labels"`
}

func ConfigFromFile(filename string) (retVal BertConfig) {
	filePath, err := filepath.Abs(filename)
	if err != nil {
		log.Fatal(err)
	}

	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	buff, err := ioutil.ReadAll(f)
	if err != nil {
		log.Fatal(err)
	}

	var config BertConfig
	err = json.Unmarshal(buff, &config)
	if err != nil {
		fmt.Println(err)
		log.Fatalf("Could not parse configuration to BertConfiguration.\n")
	}
	return config
}

// BertModel defines base architecture for BERT models.
// `Task-specific` models can be built from this base model.
// `Embeddings`: for `token`, `position` and `segment` embeddings
// `Encoder`: is a vector of layers. Each layer compose of a `self-attention`,
// an `intermedate` (linear) and an output ( linear + layer norm) sub-layers.
// `Pooler`: linear layer applied to the first element of the sequence (`[MASK]` token)
// `IsDecoder`: whether model is used as a decoder. If set to `true`
// a casual mask will be applied to hide future positions that should be attended to.
type BertModel struct {
	Embeddings BertEmbeddings
	Encoder    BertEncoder
	Pooler     BertPooler
	IsDecoder  bool
}

// NewBertModel builds a new `BertModel`
// * `p` Variable store path for the root of the BERT Model
// * `config` `BertConfig` configuration for model architecture and decoder status
func NewBertModel(p nn.Path, config BertConfig) (retVal BertModel) {
	isDecoder := false
	if config.IsDecoder {
		isDecoder = true
	}

	embeddings := NewBertEmbedding(p.Sub("embeddings"), config)
	encoder := NewBertEncoder(p.Sub("encoder"), config)
	pooler := NewBertPooler(p.Sub("pooler"), config)

	return BertModel{embeddings, encoder, pooler, isDecoder}
}

// ForwardT forwards pass through the model
//
// # Arguments
//
// * `inputIds` - Optional input tensor of shape (*batch size*, *sequenceLength*). If None, pre-computed embeddings must be provided (see `inputEmbeds`)
// * `mask` - Optional mask of shape (*batch size*, *sequenceLength*). Masked position have value 0, non-masked value 1. If None set to 1
// * `tokenTypeIds` - Optional segment id of shape (*batch size*, *sequenceLength*). Convention is value of 0 for the first sentence (incl. *[SEP]*) and 1 for the second sentence. If None set to 0.
// * `positionIds` - Optional position ids of shape (*batch size*, *sequenceLength*). If None, will be incremented from 0.
// * `inputEmbeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequenceLength*, *hiddenSize*). If None, input ids must be provided (see `inputIds`)
// * `encoderHiddenStates` - Optional encoder hidden state of shape (*batch size*, *encoderSequenceLength*, *hiddenSize*). If the model is defined as a decoder and the `encoderHiddenStates` is not None, used in the cross-attention layer as keys and values (query from the decoder).
// * `encoderMask` - Optional encoder attention mask of shape (*batch size*, *encoderSequenceLength*). If the model is defined as a decoder and the `encoderHiddenStates` is not None, used to mask encoder values. Positions with value 0 will be masked.
// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
//
// # Returns
//
// * `output` - `Tensor` of shape (*batch size*, *sequenceLength*, *hiddenSize*)
// * `pooledOutput` - `Tensor` of shape (*batch size*, *hiddenSize*)
// * `hiddenStates` - `[]ts.Tensor` of length *numHiddenLayers* with shape (*batch size*, *sequenceLength*, *hiddenSize*)
// * `attentions` - `[]ts.Tensor` of length *num_hidden_layers* with shape (*batch size*, *sequenceLength*, *hiddenSize*)
func (b BertModel) ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, encoderHiddenStates, encoderMask ts.Tensor, train bool) (retVal1, retVal2 ts.Tensor, retValOpt1, retValOpt2 []ts.Tensor, err error) {

	var (
		inputShape []int64
		device     gotch.Device
	)

	if inputIds.MustDefined() {
		if inputEmbeds.MustDefined() {
			err = fmt.Errorf("Only one of input ids or input embeddings may be set\n")
			return
		} else {
			inputShape = inputIds.MustSize()
			device = inputIds.MustDevice()
		}
	} else {
		if inputEmbeds.MustDefined() {
			size := inputEmbeds.MustSize()
			inputShape = []int64{size[0], size[1]}
			device = inputEmbeds.MustDevice()
		} else {
			err = fmt.Errorf("At least one of input ids or input embeddings must be set\n")
			return
		}
	}

	var maskTs ts.Tensor
	if mask.MustDefined() {
		maskTs = mask
	} else {
		maskTs = ts.MustOnes(inputShape, gotch.Int64, device)
	}

	var extendedAttentionMask ts.Tensor
	switch maskTs.Dim() {
	case 3:
		extendedAttentionMask = maskTs.MustUnsqueeze(1, false) // TODO: check and delete maskTs if not using later
	case 2:
		if b.IsDecoder {
			seqIds := ts.MustArange(ts.IntScalar(inputShape[1]), gotch.Float, device)
			causalMaskTmp := seqIds.MustUnsqueeze(0, false).MustUnsqueeze(0, true).MustRepeat([]int64{inputShape[0], inputShape[1], 1}, true)
			causalMask := causalMaskTmp.MustLe1(seqIds.MustUnsqueeze(0, true).MustUnsqueeze(1, true), true)
			extendedAttentionMask = causalMask.MustMatmul(mask.MustUnsqueeze(1, false).MustUnsqueeze(1, true), true)
		} else {
			extendedAttentionMask = maskTs.MustUnsqueeze(1, false).MustUnsqueeze(1, true)
		}

	default:
		err = fmt.Errorf("Invalid attention mask dimension, must be 2 or 3, got %v\n", maskTs.Dim())
	}

	extendedAttnMask := extendedAttentionMask.MustOnesLike(false).MustSub(extendedAttentionMask, true).MustMul1(ts.FloatScalar(-10000.0), true)

	// NOTE. encoderExtendedAttentionMask is an optional tensor
	var encoderExtendedAttentionMask ts.Tensor
	if b.IsDecoder && encoderHiddenStates.MustDefined() {
		size := encoderHiddenStates.MustSize()
		var encoderMaskTs ts.Tensor
		if encoderMask.MustDefined() {
			encoderMaskTs = encoderMask
		} else {
			encoderMaskTs = ts.MustOnes([]int64{size[0], size[1]}, gotch.Int64, device)
		}

		switch encoderMaskTs.Dim() {
		case 2:
			encoderExtendedAttentionMask = encoderMaskTs.MustUnsqueeze(1, true).MustUnsqueeze(1, true)
		case 3:
			encoderExtendedAttentionMask = encoderMaskTs.MustUnsqueeze(1, true)
		default:
			err = fmt.Errorf("Invalid encoder attention mask dimension, must be 2, or 3 got %v\n", encoderMaskTs.Dim())
			return
		}
	} else {
		encoderExtendedAttentionMask = ts.None
	}

	embeddingOutput, err := b.Embeddings.ForwardT(inputIds, tokenTypeIds, positionIds, inputEmbeds, train)
	if err != nil {
		return
	}

	hiddenState, allHiddenStates, allAttentions := b.Encoder.ForwardT(embeddingOutput, extendedAttnMask, encoderHiddenStates, encoderExtendedAttentionMask, train)

	pooledOutput := b.Pooler.Forward(hiddenState)

	return hiddenState, pooledOutput, allHiddenStates, allAttentions, nil
}

// BertPredictionHeadTransform:
// ============================

type BertPredictionHeadTransform struct {
	Dense      nn.Linear
	Activation common.ActivationFn
	LayerNorm  nn.LayerNorm
}

func NewBertPredictionHeadTransform(p nn.Path, config BertConfig) (retVal BertPredictionHeadTransform) {
	dense := nn.NewLinear(p.Sub("dense"), config.HiddenSize, config.HiddenSize, nn.DefaultLinearConfig())
	activation, ok := common.ActivationFnMap[config.HiddenAct]
	if !ok {
		log.Fatalf("Unsupported activation function - %v\n", config.HiddenAct)
	}

	layerNorm := nn.NewLayerNorm(p.Sub("LayerNorm"), []int64{config.HiddenSize}, nn.DefaultLayerNormConfig())

	return BertPredictionHeadTransform{dense, activation, layerNorm}
}

func (bpht BertPredictionHeadTransform) Forward(hiddenStates ts.Tensor) (retVal ts.Tensor) {
	tmp1 := hiddenStates.Apply(bpht.Dense)
	tmp2 := bpht.Activation.Fwd(tmp1)
	retVal = tmp2.Apply(bpht.LayerNorm)
	tmp1.MustDrop()
	tmp2.MustDrop()

	return retVal
}

// BertLMPredictionHead:
// =====================

type BertLMPredictionHead struct {
	Transform BertPredictionHeadTransform
	Decoder   common.LinearNoBias
	Bias      ts.Tensor
}

func NewBertLMPredictionHead(p nn.Path, config BertConfig) (retVal BertLMPredictionHead) {
	path := p.Sub("predictions")
	transform := NewBertPredictionHeadTransform(path.Sub("transform"), config)
	decoder := common.NewLinearNoBias(path.Sub("decoder"), config.HiddenSize, config.VocabSize, common.DefaultLinearNoBiasConfig())
	bias := path.NewVar("bias", []int64{config.VocabSize}, nn.NewKaimingUniformInit())

	return BertLMPredictionHead{transform, decoder, bias}
}

func (ph BertLMPredictionHead) Forward(hiddenState ts.Tensor) ts.Tensor {
	fwTensor := ph.Transform.Forward(hiddenState).Apply(ph.Decoder)

	retVal := fwTensor.MustAdd(ph.Bias, false)
	fwTensor.MustDrop()

	return retVal
}

// BertForMaskedLM:
// ================

// BertForMaskedLM is BERT for masked language model
type BertForMaskedLM struct {
	bert BertModel
	cls  BertLMPredictionHead
}

func NewBertForMaskedLM(p nn.Path, config BertConfig) (retVal BertForMaskedLM) {
	bert := NewBertModel(p.Sub("bert"), config)
	cls := NewBertLMPredictionHead(p.Sub("cls"), config)

	return BertForMaskedLM{bert, cls}
}

// ForwardT forwards pass through the model
//
// # Arguments
//
// * `inputIds` - Optional input tensor of shape (*batch size*, *sequenceLength*). If None, pre-computed embeddings must be provided (see *inputEmbeds*)
// * `mask` - Optional mask of shape (*batch size*, *sequenceLength*). Masked position have value 0, non-masked value 1. If None set to 1
// * `tokenTypeIds` -Optional segment id of shape (*batch size*, *sequenceLength*). Convention is value of 0 for the first sentence (incl. *[SEP]*) and 1 for the second sentence. If None set to 0.
// * `positionIds` - Optional position ids of shape (*batch size*, *sequenceLength*). If None, will be incremented from 0.
// * `inputEmbeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequenceLength*, *hiddenSize*). If None, input ids must be provided (see *inputIds*)
// * `encoderHiddenStates` - Optional encoder hidden state of shape (*batch size*, *encoderSequenceLength*, *hiddenSize*). If the model is defined as a decoder and the *encoderHiddenStates* is not None, used in the cross-attention layer as keys and values (query from the decoder).
// * `encoderMask` - Optional encoder attention mask of shape (*batch size*, *encoderSequenceLength*). If the model is defined as a decoder and the *encoderHiddenStates* is not None, used to mask encoder values. Positions with value 0 will be masked.
// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
//
// # Returns
//
// * `output` - `Tensor` of shape (*batch size*, *numLabels*, *vocabSize*)
// * `hiddenStates` - `[]ts.Tensor` of length *num_hidden_layers* with shape (*batch size*, *sequenceLength*, *hiddenSize*)
// * `attentions` - `[]ts.Tensor` of length *numHiddenLayers* with shape (*batch size*, *sequenceLength*, *hiddenSize*)
func (mlm BertForMaskedLM) ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, encoderHiddenStates, encoderMask ts.Tensor, train bool) (retVal1 ts.Tensor, optRetVal1, optRetVal2 []ts.Tensor) {

	hiddenState, _, allHiddenStates, allAttentions, err := mlm.bert.ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, encoderHiddenStates, encoderMask, train)
	if err != nil {
		log.Fatal(err)
	}

	predictionScores := mlm.cls.Forward(hiddenState)

	return predictionScores, allHiddenStates, allAttentions
}

// TODO: continue...
