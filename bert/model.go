package bert

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"

	"github.com/sugarme/transformer/pretrained"
	"github.com/sugarme/transformer/util"
)

// BertModel defines base architecture for BERT models.
// Task-specific models can be built from this base model.
//
// Fields:
// 	- Embeddings: for `token`, `position` and `segment` embeddings
// 	- Encoder: is a vector of layers. Each layer compose of a `self-attention`,
// an `intermedate` (linear) and an output ( linear + layer norm) sub-layers.
// 	- Pooler: linear layer applied to the first element of the sequence (`[MASK]` token)
// 	- IsDecoder: whether model is used as a decoder. If set to `true`
// a casual mask will be applied to hide future positions that should be attended to.
type BertModel struct {
	Embeddings *BertEmbeddings
	Encoder    *BertEncoder
	Pooler     *BertPooler
	IsDecoder  bool
}

// NewBertModel builds a new `BertModel`.
//
// Params:
// 	- `p`: Variable store path for the root of the BERT Model
// 	- `config`: BertConfig onfiguration for model architecture and decoder status
func NewBertModel(p nn.Path, config *BertConfig) *BertModel {
	isDecoder := false
	if config.IsDecoder {
		isDecoder = true
	}

	embeddings := NewBertEmbeddings(p.Sub("embeddings"), config)
	encoder := NewBertEncoder(p.Sub("encoder"), config)
	pooler := NewBertPooler(p.Sub("pooler"), config)

	return &BertModel{embeddings, encoder, pooler, isDecoder}
}

// ForwardT forwards pass through the model.
//
// Params:
// 	- `inputIds`: optional input tensor of shape (batch size, sequence length).
// 		If None, pre-computed embeddings must be provided (see `inputEmbeds`)
// 	- `mask`: optional mask of shape (batch size, sequence length).
// 		Masked position have value 0, non-masked value 1. If None set to 1.
// 	- `tokenTypeIds`: optional segment id of shape (batch size, sequence length).
// 		Convention is value of 0 for the first sentence (incl. [SEP]) and 1 for the second sentence. If None set to 0.
// 	- `positionIds`: optional position ids of shape (batch size, sequence length).
//		If None, will be incremented from 0.
// 	- `inputEmbeds`: optional pre-computed input embeddings of shape (batch size, sequence length, hidden size).
// 		If None, input ids must be provided (see `inputIds`).
// 	- `encoderHiddenStates`: optional encoder hidden state of shape (batch size, encoder sequence length, hidden size).
// 		If the model is defined as a decoder and the `encoderHiddenStates` is not None,
// 		used in the cross-attention layer as keys and values (query from the decoder).
// 	- `encoderMask`: optional encoder attention mask of shape (batch size, encoder sequence length).
// 		If the model is defined as a decoder and the `encoderHiddenStates` is not None, used to mask encoder values.
// 		Positions with value 0 will be masked.
// 	- `train`: boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
//
// Returns:
// 	- `output`: tensor of shape (batch size, sequence length, hidden size)
// 	- `pooledOutput`: tensor of shape (batch size, hidden size)
// 	- `hiddenStates`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
// 	- `attentions`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
func (b *BertModel) ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, encoderHiddenStates, encoderMask ts.Tensor, train bool) (retVal1, retVal2 ts.Tensor, retValOpt1, retValOpt2 []ts.Tensor, err error) {

	var (
		inputShape []int64
		device     gotch.Device
	)

	if inputIds.MustDefined() {
		if inputEmbeds.MustDefined() {
			err = fmt.Errorf("Only one of input ids or input embeddings may be set\n")
			return
		}
		inputShape = inputIds.MustSize()
		device = inputIds.MustDevice()
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
		extendedAttentionMask = maskTs.MustUnsqueeze(1, true)
	case 2:
		if b.IsDecoder {
			seqIds := ts.MustArange(ts.IntScalar(inputShape[1]), gotch.Float, device)
			causalMaskTmp := seqIds.MustUnsqueeze(0, false).MustUnsqueeze(0, true).MustRepeat([]int64{inputShape[0], inputShape[1], 1}, true)
			seqIdsTmp := seqIds.MustUnsqueeze(0, true).MustUnsqueeze(1, true)
			causalMask := causalMaskTmp.MustLe1(seqIdsTmp, true)
			seqIdsTmp.MustDrop()
			extendedAttentionMask = causalMask.MustMatmul(mask.MustUnsqueeze(1, false).MustUnsqueeze(1, true), true)
		} else {
			extendedAttentionMask = maskTs.MustUnsqueeze(1, true).MustUnsqueeze(1, true)
		}
	default:
		err = fmt.Errorf("Invalid attention mask dimension, must be 2 or 3, got %v\n", maskTs.Dim())
	}

	mul1 := ts.FloatScalar(-10000.0)
	extendedAttnMask := extendedAttentionMask.MustOnesLike(false).MustSub(extendedAttentionMask, true).MustMul1(mul1, true)
	mul1.MustDrop()
	extendedAttentionMask.MustDrop()

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

	extendedAttnMask.MustDrop()
	// embeddingOutput.MustDrop() // NOTE. free up this tensor causes panic after some cycles of forwarding. Why?
	encoderExtendedAttentionMask.MustDrop()

	return hiddenState, pooledOutput, allHiddenStates, allAttentions, nil
}

// BertPredictionHeadTransform:
// ============================

// BertPredictionHeadTransform holds layers of BERT prediction head transform.
type BertPredictionHeadTransform struct {
	Dense      *nn.Linear
	Activation util.ActivationFn
	LayerNorm  *nn.LayerNorm
}

// NewBertPredictionHead creates BertPredictionHeadTransform.
func NewBertPredictionHeadTransform(p nn.Path, config *BertConfig) *BertPredictionHeadTransform {
	dense := nn.NewLinear(p.Sub("dense"), config.HiddenSize, config.HiddenSize, nn.DefaultLinearConfig())
	activation, ok := util.ActivationFnMap[config.HiddenAct]
	if !ok {
		log.Fatalf("Unsupported activation function - %v\n", config.HiddenAct)
	}

	layerNorm := nn.NewLayerNorm(p.Sub("LayerNorm"), []int64{config.HiddenSize}, nn.DefaultLayerNormConfig())

	return &BertPredictionHeadTransform{&dense, activation, &layerNorm}
}

// Forward forwards through the model.
func (bpht *BertPredictionHeadTransform) Forward(hiddenStates ts.Tensor) (retVal ts.Tensor) {
	tmp1 := hiddenStates.Apply(bpht.Dense)
	tmp2 := bpht.Activation.Fwd(tmp1)
	retVal = tmp2.Apply(bpht.LayerNorm)
	tmp1.MustDrop()
	tmp2.MustDrop()

	return retVal
}

// BertLMPredictionHead:
// =====================

// BertLMPredictionHead constructs layers for BERT prediction head.
type BertLMPredictionHead struct {
	Transform *BertPredictionHeadTransform
	Decoder   *util.LinearNoBias
	Bias      ts.Tensor
}

// NewBertLMPredictionHead creates BertLMPredictionHead.
func NewBertLMPredictionHead(p nn.Path, config *BertConfig) *BertLMPredictionHead {
	path := p.Sub("predictions")
	transform := NewBertPredictionHeadTransform(path.Sub("transform"), config)
	decoder := util.NewLinearNoBias(path.Sub("decoder"), config.HiddenSize, config.VocabSize, util.DefaultLinearNoBiasConfig())
	bias := path.NewVar("bias", []int64{config.VocabSize}, nn.NewKaimingUniformInit())

	return &BertLMPredictionHead{transform, decoder, bias}
}

// Forward fowards through the model.
func (ph *BertLMPredictionHead) Forward(hiddenState ts.Tensor) ts.Tensor {
	fwTensor := ph.Transform.Forward(hiddenState).Apply(ph.Decoder)

	retVal := fwTensor.MustAdd(ph.Bias, false)
	fwTensor.MustDrop()

	return retVal
}

// BertForMaskedLM:
// ================

// BertForMaskedLM is BERT for masked language model
type BertForMaskedLM struct {
	bert *BertModel
	cls  *BertLMPredictionHead
}

// NewBertForMaskedLM creates BertForMaskedLM.
func NewBertForMaskedLM(p nn.Path, config *BertConfig) *BertForMaskedLM {
	bert := NewBertModel(p.Sub("bert"), config)
	cls := NewBertLMPredictionHead(p.Sub("cls"), config)

	return &BertForMaskedLM{bert, cls}
}

// Load loads model from file or model name. It also updates
// default configuration parameters if provided.
// This method implements `PretrainedModel` interface.
func (mlm *BertForMaskedLM) Load(modelNameOrPath string, config interface{ pretrained.Config }, params map[string]interface{}, vs nn.VarStore) error {
	var urlOrFilename string
	// If modelName, infer to default configuration filename:
	if modelFile, ok := pretrained.BertModels[modelNameOrPath]; ok {
		urlOrFilename = modelFile
	} else {
		// Otherwise, just take the input
		urlOrFilename = modelNameOrPath
	}

	cachedFile, err := util.CachedPath(urlOrFilename)
	if err != nil {
		return err
	}

	p := vs.Root()
	mlm.bert = NewBertModel(p.Sub("bert"), config.(*BertConfig))
	mlm.cls = NewBertLMPredictionHead(p.Sub("cls"), config.(*BertConfig))
	err = vs.Load(cachedFile)
	if err != nil {
		log.Fatalf("Load model weight error: \n%v", err)
	}

	return nil
}

// ForwardT forwards pass through the model.
//
// Params:
// 	- `inputIds`: optional input tensor of shape (batch size, sequence length).
// 		If None, pre-computed embeddings must be provided (see `inputEmbeds`)
// 	- `mask`: optional mask of shape (batch size, sequence length).
// 		Masked position have value 0, non-masked value 1. If None set to 1.
// 	- `tokenTypeIds`: optional segment id of shape (batch size, sequence length).
// 		Convention is value of 0 for the first sentence (incl. [SEP]) and 1 for the second sentence. If None set to 0.
// 	- `positionIds`: optional position ids of shape (batch size, sequence length).
//		If None, will be incremented from 0.
// 	- `inputEmbeds`: optional pre-computed input embeddings of shape (batch size, sequence length, hidden size).
// 		If None, input ids must be provided (see `inputIds`).
// 	- `encoderHiddenStates`: optional encoder hidden state of shape (batch size, encoder sequence length, hidden size).
// 		If the model is defined as a decoder and the `encoderHiddenStates` is not None,
// 		used in the cross-attention layer as keys and values (query from the decoder).
// 	- `encoderMask`: optional encoder attention mask of shape (batch size, encoder sequence length).
// 		If the model is defined as a decoder and the `encoderHiddenStates` is not None, used to mask encoder values.
// 		Positions with value 0 will be masked.
// 	- `train`: boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
//
// Returns:
// 	- `output`: tensor of shape (batch size, sequence length, hidden size)
// 	- `hiddenStates`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
// 	- `attentions`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
func (mlm *BertForMaskedLM) ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, encoderHiddenStates, encoderMask ts.Tensor, train bool) (retVal1 ts.Tensor, optRetVal1, optRetVal2 []ts.Tensor) {

	hiddenState, _, allHiddenStates, allAttentions, err := mlm.bert.ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, encoderHiddenStates, encoderMask, train)
	if err != nil {
		log.Fatal(err)
	}

	predictionScores := mlm.cls.Forward(hiddenState)

	return predictionScores, allHiddenStates, allAttentions
}

// BERT for sequence classification:
// =================================

// BertForSequenceClassification is Base BERT model with a classifier head to perform
// sentence or document-level classification.
//
// It is made of the following blocks:
// 	- `bert`: Base BertModel
// 	- `classifier`: BERT linear layer for classification
type BertForSequenceClassification struct {
	bert       *BertModel
	dropout    *util.Dropout
	classifier *nn.Linear
}

// NewBertForSequenceClassification creates a new `BertForSequenceClassification`.
//
// Params:
// 	- `p`: ariable store path for the root of the BertForSequenceClassification model
// 	- `config`: `BertConfig` object defining the model architecture and number of classes
//
// Example:
//
// 		device := gotch.CPU
// 		vs := nn.NewVarStore(device)
// 		config := bert.ConfigFromFile("path/to/config.json")
// 		p := vs.Root()
// 		bert := NewBertForSequenceClassification(p.Sub("bert"), config)
func NewBertForSequenceClassification(p nn.Path, config *BertConfig) *BertForSequenceClassification {
	bert := NewBertModel(p.Sub("bert"), config)
	dropout := util.NewDropout(config.HiddenDropoutProb)
	numLabels := len(config.Id2Label)

	classifier := nn.NewLinear(p.Sub("classifier"), config.HiddenSize, int64(numLabels), nn.DefaultLinearConfig())

	return &BertForSequenceClassification{
		bert:       bert,
		dropout:    dropout,
		classifier: &classifier,
	}
}

// ForwardT forwards pass through the model.
//
// Params:
// 	- `inputIds`: optional input tensor of shape (batch size, sequence length).
// 		If None, pre-computed embeddings must be provided (see `inputEmbeds`)
// 	- `mask`: optional mask of shape (batch size, sequence length).
// 		Masked position have value 0, non-masked value 1. If None set to 1.
// 	- `tokenTypeIds`: optional segment id of shape (batch size, sequence length).
// 		Convention is value of 0 for the first sentence (incl. [SEP]) and 1 for the second sentence. If None set to 0.
// 	- `positionIds`: optional position ids of shape (batch size, sequence length).
//		If None, will be incremented from 0.
// 	- `inputEmbeds`: optional pre-computed input embeddings of shape (batch size, sequence length, hidden size).
// 		If None, input ids must be provided (see `inputIds`).
// 	- `encoderHiddenStates`: optional encoder hidden state of shape (batch size, encoder sequence length, hidden size).
// 		If the model is defined as a decoder and the `encoderHiddenStates` is not None,
// 		used in the cross-attention layer as keys and values (query from the decoder).
// 	- `encoderMask`: optional encoder attention mask of shape (batch size, encoder sequence length).
// 		If the model is defined as a decoder and the `encoderHiddenStates` is not None, used to mask encoder values.
// 		Positions with value 0 will be masked.
// 	- `train`: boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
//
// Returns:
// 	- `output`: tensor of shape (batch size, sequence length, hidden size)
// 	- `pooledOutput`: tensor of shape (batch size, hidden size)
// 	- `hiddenStates`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
// 	- `attentions`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
func (bsc *BertForSequenceClassification) ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (retVal ts.Tensor, retValOpt1, retValOpt2 []ts.Tensor) {
	_, pooledOutput, allHiddenStates, allAttentions, err := bsc.bert.ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, ts.None, ts.None, train)

	if err != nil {
		log.Fatalf("call bert.ForwardT error: %v", err)
	}

	dropoutOutput := pooledOutput.ApplyT(bsc.dropout, train)

	output := dropoutOutput.Apply(bsc.classifier)
	dropoutOutput.MustDrop()

	return output, allHiddenStates, allAttentions
}

// BERT for multiple choices :
// ===========================

// BertForMultipleChoice constructs multiple choices model using a BERT base model and a linear classifier.
// Input should be in the form `[CLS] Context [SEP] Possible choice [SEP]`. The choice is made along the batch axis,
// assuming all elements of the batch are alternatives to be chosen from for a given context.
//
// It is made of the following blocks:
// 	- `bert`: Base BertModel
// 	- `classifier`: Linear layer for multiple choices
type BertForMultipleChoice struct {
	bert       *BertModel
	dropout    *util.Dropout
	classifier *nn.Linear
}

// NewBertForMultipleChoice creates a new `BertForMultipleChoice`.
//
// Params:
// 	- `p`: Variable store path for the root of the BertForMultipleChoice model
// 	- `config`: `BertConfig` object defining the model architecture
func NewBertForMultipleChoice(p nn.Path, config *BertConfig) *BertForMultipleChoice {
	bert := NewBertModel(p.Sub("bert"), config)
	dropout := util.NewDropout(config.HiddenDropoutProb)
	classifier := nn.NewLinear(p.Sub("classifier"), config.HiddenSize, 1, nn.DefaultLinearConfig())

	return &BertForMultipleChoice{
		bert:       bert,
		dropout:    dropout,
		classifier: &classifier,
	}
}

// ForwardT forwards pass through the model.
//
// Params:
// 	- `inputIds`: optional input tensor of shape (batch size, sequence length).
// 		If None, pre-computed embeddings must be provided (see `inputEmbeds`)
// 	- `mask`: optional mask of shape (batch size, sequence length).
// 		Masked position have value 0, non-masked value 1. If None set to 1.
// 	- `tokenTypeIds`: optional segment id of shape (batch size, sequence length).
// 		Convention is value of 0 for the first sentence (incl. [SEP]) and 1 for the second sentence. If None set to 0.
// 	- `positionIds`: optional position ids of shape (batch size, sequence length).
//		If None, will be incremented from 0.
// 	- `train`: boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
//
// Returns:
// 	- `output`: tensor of shape (batch size, sequence length, hidden size)
// 	- `hiddenStates`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
// 	- `attentions`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
func (mc *BertForMultipleChoice) ForwardT(inputIds, mask, tokenTypeIds, positionIds ts.Tensor, train bool) (retVal ts.Tensor, retValOpt1, retValOpt2 []ts.Tensor) {

	inputIdsSize := inputIds.MustSize()
	fmt.Printf("inputIdsSize: %v\n", inputIdsSize)
	numChoices := inputIdsSize[1]
	inputIdsView := inputIds.MustView([]int64{-1, inputIdsSize[len(inputIdsSize)-1]}, false)

	maskView := ts.None
	if mask.MustDefined() {
		maskSize := mask.MustSize()
		maskView = mask.MustView([]int64{-1, maskSize[len(maskSize)-1]}, false)
	}

	tokenTypeIdsView := ts.None
	if tokenTypeIds.MustDefined() {
		tokenTypeIdsSize := tokenTypeIds.MustSize()
		tokenTypeIdsView = tokenTypeIds.MustView([]int64{-1, tokenTypeIdsSize[len(tokenTypeIdsSize)-1]}, false)
	}

	positionIdsView := ts.None
	if positionIds.MustDefined() {
		positionIdsSize := positionIds.MustSize()
		positionIdsView = positionIds.MustView([]int64{-1, positionIdsSize[len(positionIdsSize)-1]}, false)
	}

	_, pooledOutput, allHiddenStates, allAttentions, err := mc.bert.ForwardT(inputIdsView, maskView, tokenTypeIdsView, positionIdsView, ts.None, ts.None, ts.None, train)
	if err != nil {
		log.Fatalf("Call 'BertForMultipleChoice ForwordT' method error: %v\n", err)
	}

	outputDropout := pooledOutput.ApplyT(mc.dropout, train)
	outputClassifier := outputDropout.Apply(mc.classifier)

	output := outputClassifier.MustView([]int64{-1, numChoices}, false)

	outputDropout.MustDrop()
	outputClassifier.MustDrop()

	return output, allHiddenStates, allAttentions
}

// BERT for token classification (e.g., NER, POS):
// ===============================================

// BertForTokenClassification constructs token-level classifier predicting a label for each token provided.
// Note that because of wordpiece tokenization, the labels predicted are not necessarily aligned with words in the sentence.
//
// It is made of the following blocks:
// 	- `bert`: Base BertModel
// 	- `classifier`: Linear layer for token classification
type BertForTokenClassification struct {
	bert       *BertModel
	dropout    *util.Dropout
	classifier *nn.Linear
}

// NewBertForTokenClassification creates a new `BertForTokenClassification`
//
// Params:
// 	- `p`: Variable store path for the root of the BertForTokenClassification model
// 	- `config`: `BertConfig` object defining the model architecture, number of output labels and label mapping
func NewBertForTokenClassification(p nn.Path, config *BertConfig) *BertForTokenClassification {
	bert := NewBertModel(p.Sub("bert"), config)
	dropout := util.NewDropout(config.HiddenDropoutProb)

	numLabels := len(config.Id2Label)
	classifier := nn.NewLinear(p.Sub("classifier"), config.HiddenSize, int64(numLabels), nn.DefaultLinearConfig())

	return &BertForTokenClassification{
		bert:       bert,
		dropout:    dropout,
		classifier: &classifier,
	}
}

// ForwordT forwards pass through the model.
//
// Params:
// 	- `inputIds`: optional input tensor of shape (batch size, sequence length).
// 		If None, pre-computed embeddings must be provided (see `inputEmbeds`)
// 	- `mask`: optional mask of shape (batch size, sequence length).
// 		Masked position have value 0, non-masked value 1. If None set to 1.
// 	- `tokenTypeIds`: optional segment id of shape (batch size, sequence length).
// 		Convention is value of 0 for the first sentence (incl. [SEP]) and 1 for the second sentence. If None set to 0.
// 	- `positionIds`: optional position ids of shape (batch size, sequence length).
//		If None, will be incremented from 0.
// 	- `inputEmbeds`: optional pre-computed input embeddings of shape (batch size, sequence length, hidden size).
// 		If None, input ids must be provided (see `inputIds`).
// 	- `train`: boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
//
// Returns:
// 	- `output`: tensor of shape (batch size, sequence length, hidden size)
// 	- `hiddenStates`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
// 	- `attentions`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
func (tc *BertForTokenClassification) ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (retVal ts.Tensor, retValOpt1, retValOpt2 []ts.Tensor) {

	hiddenState, _, allHiddenStates, allAttentions, err := tc.bert.ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, ts.None, ts.None, train)
	if err != nil {
		log.Fatalf("Call 'BertForTokenClassification ForwardT' method error: %v\n", err)
	}

	outputDropout := hiddenState.ApplyT(tc.dropout, train)
	output := outputDropout.Apply(tc.classifier)

	outputDropout.MustDrop()

	return output, allHiddenStates, allAttentions
}

// BERT for question answering:
// ============================

// BertForQuestionAnswering constructs extractive question-answering model based on a BERT language model. Identifies the segment of a context that answers a provided question.
//
// Please note that a significant amount of pre- and post-processing is required to perform end-to-end question answering.
// See the question answering pipeline (also provided in this crate) for more details.
//
// It is made of the following blocks:
// 	- `bert`: Base BertModel
// 	- `qa_outputs`: Linear layer for question answering
type BertForQuestionAnswering struct {
	bert      *BertModel
	qaOutputs *nn.Linear
}

// NewBertForQuestionAnswering creates a new `BertForQuestionAnswering`.
//
// Params:
// 	- `p`: Variable store path for the root of the BertForQuestionAnswering model
// 	- `config`: `BertConfig` object defining the model architecture
func NewBertForQuestionAnswering(p nn.Path, config *BertConfig) *BertForQuestionAnswering {
	bert := NewBertModel(p.Sub("bert"), config)

	numLabels := 2
	qaOutputs := nn.NewLinear(p.Sub("qa_outputs"), config.HiddenSize, int64(numLabels), nn.DefaultLinearConfig())

	return &BertForQuestionAnswering{
		bert:      bert,
		qaOutputs: &qaOutputs,
	}
}

// Load loads model from file or model name. It also updates
// default configuration parameters if provided.
// This method implements `PretrainedModel` interface.
func (qa *BertForQuestionAnswering) Load(modelNameOrPath string, config interface{ pretrained.Config }, params map[string]interface{}, vs nn.VarStore) error {
	var urlOrFilename string
	// If modelName, infer to default configuration filename:
	if modelFile, ok := pretrained.BertModels[modelNameOrPath]; ok {
		urlOrFilename = modelFile
	} else {
		// Otherwise, just take the input
		urlOrFilename = modelNameOrPath
	}

	cachedFile, err := util.CachedPath(urlOrFilename)
	if err != nil {
		return err
	}

	p := vs.Root()

	bert := NewBertModel(p.Sub("bert"), config.(*BertConfig))

	numLabels := 2
	qaOutputs := nn.NewLinear(p.Sub("qa_outputs"), config.(*BertConfig).HiddenSize, int64(numLabels), nn.DefaultLinearConfig())

	qa.bert = bert
	qa.qaOutputs = &qaOutputs
	err = vs.Load(cachedFile)
	if err != nil {
		log.Fatalf("Load model weight error: \n%v", err)
	}

	return nil
}

// ForwardT forwards pass through the model.
//
// Params:
// 	- `inputIds`: optional input tensor of shape (batch size, sequence length).
// 		If None, pre-computed embeddings must be provided (see `inputEmbeds`)
// 	- `mask`: optional mask of shape (batch size, sequence length).
// 		Masked position have value 0, non-masked value 1. If None set to 1.
// 	- `tokenTypeIds`: optional segment id of shape (batch size, sequence length).
// 		Convention is value of 0 for the first sentence (incl. [SEP]) and 1 for the second sentence. If None set to 0.
// 	- `positionIds`: optional position ids of shape (batch size, sequence length).
//		If None, will be incremented from 0.
// 	- `inputEmbeds`: optional pre-computed input embeddings of shape (batch size, sequence length, hidden size).
// 		If None, input ids must be provided (see `inputIds`).
// 	- `train`: boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
//
// Returns:
// 	- `output`: tensor of shape (batch size, sequence length, hidden size)
// 	- `hiddenStates`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
// 	- `attentions`: slice of tensors of length numHiddenLayers with shape (batch size, sequenceLength, hiddenSize)
func (qa *BertForQuestionAnswering) ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (retVal1, retVal2 ts.Tensor, retValOpt1, retValOpt2 []ts.Tensor, err error) {

	hiddenState, pooledOutput, allHiddenStates, allAttentions, err := qa.bert.ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, ts.None, ts.None, train)

	pooledOutput.MustDrop() // don't use this. But need to clear memory in C land.
	if err != nil {
		// log.Fatalf("Call 'BertForTokenClassification ForwardT' method error: %v\n", err)
		return ts.None, ts.None, nil, nil, err
	}

	sequenceOutput := hiddenState.Apply(qa.qaOutputs)
	logits := sequenceOutput.MustSplit(1, -1, true) // -1 : split along last size
	startLogits := logits[0].MustSqueeze1(int64(-1), false)
	endLogits := logits[1].MustSqueeze1(int64(-1), false)

	for _, logitTs := range logits {
		logitTs.MustDrop()
	}

	hiddenState.MustDrop()

	return startLogits, endLogits, allHiddenStates, allAttentions, nil
}
