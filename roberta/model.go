// roberta package implements Roberta transformer model.
package roberta

import (
	// "fmt"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"

	"github.com/sugarme/transformer/bert"
	"github.com/sugarme/transformer/pretrained"
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

// Load loads model from file or model name. It also updates
// default configuration parameters if provided.
// This method implements `PretrainedModel` interface.
func (mlm *RobertaForMaskedLM) Load(modelNameOrPath string, config interface{ pretrained.Config }, params map[string]interface{}, vs nn.VarStore) error {
	var urlOrFilename string
	// If modelName, infer to default configuration filename:
	if modelFile, ok := pretrained.RobertaModels[modelNameOrPath]; ok {
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

	mlm.roberta = bert.NewBertModel(p.Sub("roberta"), config.(*bert.BertConfig))
	mlm.lmHead = NewRobertaLMHead(p.Sub("lm_head"), config.(*bert.BertConfig))

	err = vs.Load(cachedFile)
	if err != nil {
		return err
	}

	return nil
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

// RoberatClassificationHead holds data for Roberta classification head.
type RobertaClassificationHead struct {
	dense   *nn.Linear
	dropout *util.Dropout
	outProj *nn.Linear
}

// NewRobertaClassificationHead create a new RobertaClassificationHead.
func NewRobertaClassificationHead(p nn.Path, config *bert.BertConfig) *RobertaClassificationHead {
	dense := nn.NewLinear(p.Sub("dense"), config.HiddenSize, config.HiddenSize, nn.DefaultLinearConfig())
	numLabels := int64(len(config.Id2Label))
	outProj := nn.NewLinear(p.Sub("out_proj"), config.HiddenSize, numLabels, nn.DefaultLinearConfig())
	dropout := util.NewDropout(config.HiddenDropoutProb)

	return &RobertaClassificationHead{
		dense:   &dense,
		dropout: dropout,
		outProj: &outProj,
	}
}

// ForwardT forwards pass through model.
func (ch *RobertaClassificationHead) ForwardT(hiddenStates ts.Tensor, train bool) ts.Tensor {
	appliedDO1 := hiddenStates.MustSelect(1, 0, false).ApplyT(ch.dropout, train)
	appliedDense := appliedDO1.Apply(ch.dense)
	tanhTs := appliedDense.MustTanh(false)
	appliedDO2 := tanhTs.ApplyT(ch.dropout, train)
	retVal := appliedDO2.Apply(ch.outProj)

	appliedDO1.MustDrop()
	appliedDense.MustDrop()
	tanhTs.MustDrop()
	appliedDO2.MustDrop()

	return retVal
}

// RobertaForSequenceClassification holds data for Roberta sequence classification model.
// It's used for performing sentence or document-level classification.
type RobertaForSequenceClassification struct {
	roberta    *bert.BertModel
	classifier *RobertaClassificationHead
}

// NewRobertaForSequenceClassification creates a new RobertaForSequenceClassification model.
func NewRobertaForSequenceClassification(p nn.Path, config *bert.BertConfig) *RobertaForSequenceClassification {
	roberta := bert.NewBertModel(p.Sub("roberta"), config)
	classifier := NewRobertaClassificationHead(p.Sub("classifier"), config)

	return &RobertaForSequenceClassification{
		roberta:    roberta,
		classifier: classifier,
	}
}

// Load loads model from file or model name. It also updates default configuration parameters if provided.
//
// This method implements `PretrainedModel` interface.
func (sc *RobertaForSequenceClassification) Load(modelNameOrPath string, config interface{ pretrained.Config }, params map[string]interface{}, vs nn.VarStore) error {
	var urlOrFilename string
	// If modelName, infer to default configuration filename:
	if modelFile, ok := pretrained.RobertaModels[modelNameOrPath]; ok {
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

	sc.roberta = bert.NewBertModel(p.Sub("roberta"), config.(*bert.BertConfig))
	sc.classifier = NewRobertaClassificationHead(p.Sub("classifier"), config.(*bert.BertConfig))

	err = vs.Load(cachedFile)
	if err != nil {
		return err
	}

	return nil
}

// Forward forwards pass through the model.
func (sc *RobertaForSequenceClassification) ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (labels ts.Tensor, hiddenStates, attentions []ts.Tensor, err error) {

	hiddenState, _, hiddenStates, attentions, err := sc.roberta.ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, ts.None, ts.None, train)
	if err != nil {
		return ts.None, nil, nil, err
	}

	labels = sc.classifier.ForwardT(hiddenState, train)
	hiddenState.MustDrop()

	return labels, hiddenStates, attentions, nil
}

// RobertaForMultipleChoice holds data for Roberta multiple choice model.
//
// Input should be in form of `<s> Context </s> Possible choice </s>`.
// The choice is made along the batch axis, assuming all elements of the batch are
// alternatives to be chosen from for a given context.
type RobertaForMultipleChoice struct {
	roberta    *bert.BertModel
	dropout    *util.Dropout
	classifier *nn.Linear
}

// NewRobertaForMultipleChoice creates a new RobertaForMultipleChoice model.
func NewRobertaForMultipleChoice(p nn.Path, config *bert.BertConfig) *RobertaForMultipleChoice {
	roberta := bert.NewBertModel(p.Sub("roberta"), config)
	dropout := util.NewDropout(config.HiddenDropoutProb)
	classifier := nn.NewLinear(p.Sub("classifier"), config.HiddenSize, 1, nn.DefaultLinearConfig())

	return &RobertaForMultipleChoice{
		roberta:    roberta,
		dropout:    dropout,
		classifier: &classifier,
	}
}

// Load loads model from file or model name. It also updates default configuration parameters if provided.
//
// This method implements `PretrainedModel` interface.
func (mc *RobertaForMultipleChoice) Load(modelNameOrPath string, config interface{ pretrained.Config }, params map[string]interface{}, vs nn.VarStore) error {
	var urlOrFilename string
	// If modelName, infer to default configuration filename:
	if modelFile, ok := pretrained.RobertaModels[modelNameOrPath]; ok {
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

	mc.roberta = bert.NewBertModel(p.Sub("roberta"), config.(*bert.BertConfig))
	mc.dropout = util.NewDropout(config.(*bert.BertConfig).HiddenDropoutProb)
	classifier := nn.NewLinear(p.Sub("classifier"), config.(*bert.BertConfig).HiddenSize, 1, nn.DefaultLinearConfig())
	mc.classifier = &classifier

	err = vs.Load(cachedFile)
	if err != nil {
		return err
	}

	return nil
}

// ForwardT forwards pass through the model.
func (mc *RobertaForMultipleChoice) ForwardT(inputIds, mask, tokenTypeIds, positionIds ts.Tensor, train bool) (output ts.Tensor, hiddenStates, attentions []ts.Tensor, err error) {

	numChoices := inputIds.MustSize()[1]

	inputIdsSize := inputIds.MustSize()
	flatInputIds := inputIds.MustView([]int64{-1, inputIdsSize[len(inputIdsSize)-1]}, false)

	flatPositionIds := ts.None
	if positionIds.MustDefined() {
		positionIdsSize := positionIds.MustSize()
		flatPositionIds = positionIds.MustView([]int64{-1, positionIdsSize[len(positionIdsSize)-1]}, false)
	}

	flatTokenTypeIds := ts.None
	if tokenTypeIds.MustDefined() {
		tokenTypeIdsSize := tokenTypeIds.MustSize()
		flatTokenTypeIds = tokenTypeIds.MustView([]int64{-1, tokenTypeIdsSize[len(tokenTypeIdsSize)-1]}, false)
	}

	flatMask := ts.None
	if mask.MustDefined() {
		flatMaskSize := flatMask.MustSize()
		flatMask = mask.MustView([]int64{-1, flatMaskSize[len(flatMaskSize)-1]}, false)
	}

	var pooledOutput ts.Tensor
	_, pooledOutput, hiddenStates, attentions, err = mc.roberta.ForwardT(flatInputIds, flatMask, flatTokenTypeIds, flatPositionIds, ts.None, ts.None, ts.None, train)
	if err != nil {
		return ts.None, nil, nil, err
	}

	appliedDO := pooledOutput.ApplyT(mc.dropout, train)
	appliedCls := appliedDO.Apply(mc.classifier)
	output = appliedCls.MustView([]int64{-1, numChoices}, true)

	appliedDO.MustDrop()

	return output, hiddenStates, attentions, nil
}

// RobertaForTokenClassification holds data for Roberta token classification model.
type RobertaForTokenClassification struct {
	roberta    *bert.BertModel
	dropout    *util.Dropout
	classifier *nn.Linear
}

// NewRobertaForTokenClassification creates a new RobertaForTokenClassification model.
func NewRobertaForTokenClassification(p nn.Path, config *bert.BertConfig) *RobertaForTokenClassification {
	roberta := bert.NewBertModel(p.Sub("roberta"), config)
	dropout := util.NewDropout(config.HiddenDropoutProb)
	numLabels := int64(len(config.Id2Label))
	classifier := nn.NewLinear(p.Sub("classifier"), config.HiddenSize, numLabels, nn.DefaultLinearConfig())

	return &RobertaForTokenClassification{
		roberta:    roberta,
		dropout:    dropout,
		classifier: &classifier,
	}
}

// Load loads model from file or model name. It also updates default configuration parameters if provided.
//
// This method implements `PretrainedModel` interface.
func (tc *RobertaForTokenClassification) Load(modelNameOrPath string, config interface{ pretrained.Config }, params map[string]interface{}, vs nn.VarStore) error {
	var urlOrFilename string
	// If modelName, infer to default configuration filename:
	if modelFile, ok := pretrained.RobertaModels[modelNameOrPath]; ok {
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

	roberta := bert.NewBertModel(p.Sub("roberta"), config.(*bert.BertConfig))
	dropout := util.NewDropout(config.(*bert.BertConfig).HiddenDropoutProb)
	numLabels := int64(len(config.(*bert.BertConfig).Id2Label))
	classifier := nn.NewLinear(p.Sub("classifier"), config.(*bert.BertConfig).HiddenSize, numLabels, nn.DefaultLinearConfig())

	tc.roberta = roberta
	tc.dropout = dropout
	tc.classifier = &classifier

	err = vs.Load(cachedFile)
	if err != nil {
		return err
	}

	return nil
}

// ForwardT forwards pass through the model.
func (tc *RobertaForTokenClassification) ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (output ts.Tensor, hiddenStates, attentions []ts.Tensor, err error) {
	hiddenState, _, hiddenStates, attentions, err := tc.roberta.ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, ts.None, ts.None, train)
	if err != nil {
		return ts.None, nil, nil, err
	}

	appliedDO := hiddenState.ApplyT(tc.dropout, train)
	output = appliedDO.Apply(tc.classifier)

	appliedDO.MustDrop()

	return output, hiddenStates, attentions, nil
}

// RobertaForQuestionAnswering constructs layers for Roberta question answering model.
type RobertaForQuestionAnswering struct {
	roberta   *bert.BertModel
	qaOutputs *nn.Linear
}

// NewRobertaQuestionAnswering creates a new RobertaForQuestionAnswering model.
func NewRobertaForQuestionAnswering(p nn.Path, config *bert.BertConfig) *RobertaForQuestionAnswering {
	roberta := bert.NewBertModel(p.Sub("roberta"), config)
	numLabels := int64(2)
	qaOutputs := nn.NewLinear(p.Sub("qa_outputs"), config.HiddenSize, numLabels, nn.DefaultLinearConfig())

	return &RobertaForQuestionAnswering{
		roberta:   roberta,
		qaOutputs: &qaOutputs,
	}
}

// Load loads model from file or model name. It also updates default configuration parameters if provided.
//
// This method implements `PretrainedModel` interface.
func (qa *RobertaForQuestionAnswering) Load(modelNameOrPath string, config interface{ pretrained.Config }, params map[string]interface{}, vs nn.VarStore) error {
	var urlOrFilename string
	// If modelName, infer to default configuration filename:
	if modelFile, ok := pretrained.RobertaModels[modelNameOrPath]; ok {
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

	roberta := bert.NewBertModel(p.Sub("roberta"), config.(*bert.BertConfig))
	numLabels := int64(2)
	qaOutputs := nn.NewLinear(p.Sub("qa_outputs"), config.(*bert.BertConfig).HiddenSize, numLabels, nn.DefaultLinearConfig())

	qa.roberta = roberta
	qa.qaOutputs = &qaOutputs

	err = vs.Load(cachedFile)
	if err != nil {
		return err
	}

	return nil
}

// ForwadT forwards pass through the model.
func (qa *RobertaForQuestionAnswering) ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (startScores, endScores ts.Tensor, hiddenStates, attentions []ts.Tensor, err error) {
	hiddenState, _, hiddenStates, attentions, err := qa.roberta.ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds, ts.None, ts.None, train)
	if err != nil {
		return ts.None, ts.None, nil, nil, err
	}

	sequenceOutput := hiddenState.Apply(qa.qaOutputs)
	logits := sequenceOutput.MustSplit(1, -1, true)
	startScores = logits[0].MustSqueeze1(-1, false)
	endScores = logits[1].MustSqueeze1(-1, false)

	for _, x := range logits {
		x.MustDrop()
	}

	return startScores, endScores, hiddenStates, attentions, nil
}
