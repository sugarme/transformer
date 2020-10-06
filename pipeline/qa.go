package pipeline

import (
	"fmt"
	"log"
	"reflect"
	"strings"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/tokenizer"

	"github.com/sugarme/transformer"
	"github.com/sugarme/transformer/bert"
	"github.com/sugarme/transformer/pretrained"
	"github.com/sugarme/transformer/roberta"
)

// Question Answering pipeline
// Extractive question answering from a given question and context. By default, the dependencies for this
// model will be downloaded for a DistilBERT model finetuned on SQuAD (Stanford Question Answering Dataset).
// Customized DistilBERT models can be loaded by overwriting the resources in the configuration.
// The dependencies will be downloaded to the user's home directory, under ~/.cache/transformer/distilbert-qa

type QAInput struct {
	Question string
	Context  string
}

type QAExample struct {
	Question         string
	Context          string
	DocTokens        []string
	CharToWordOffset []int
}

type QAFeature struct {
	InputIds         []int64
	AttentionMask    []int64
	TokenToOriginMap map[int]int
	PMask            []int8
	ExampleIndex     int
}

// Answer is output for question answering.
type Answer struct {
	Score  float64
	Start  int // start position of answer span
	End    int // end position of answer span
	Answer string
}

func removeDuplicates(items []interface{}) []interface{} {
	keys := make(map[interface{}]bool)
	list := []interface{}{}
	for _, item := range items {
		if _, value := keys[item]; !value {
			keys[item] = true
			list = append(list, item)
		}
	}
	return list
}

func NewQAExample(question string, context string) *QAExample {

	docTokens, charToWordOffset := splitContext(context)

	return &QAExample{
		Question:         question,
		Context:          context,
		DocTokens:        docTokens,
		CharToWordOffset: charToWordOffset,
	}
}

func splitContext(context string) ([]string, []int) {
	var docTokens []string
	var charToWordOffset []int
	var currentWord []rune
	var previousWhiteSpace bool = false

	for _, char := range context {
		charToWordOffset = append(charToWordOffset, len([]byte(string(char))))
		if isWhiteSpace(char) {
			previousWhiteSpace = true
			if len(currentWord) > 0 {
				docTokens = append(docTokens, string(currentWord))
			}
		} else {
			if previousWhiteSpace {
				currentWord = nil
			}

			currentWord = append(currentWord, char)
			previousWhiteSpace = false
		}
	}

	// Last word
	if len(currentWord) > 0 {
		docTokens = append(docTokens, string(currentWord))
	}

	return docTokens, charToWordOffset
}

func isWhiteSpace(char rune) bool {
	if char == ' ' || char == '\t' || char == '\r' || char == '\n' || char == 0x202F {
		return true
	}
	return false
}

// QuestionAnsweringConfig holds configuration for question answering
type QuestionAnsweringConfig struct {
	ModelNameOrPath  string
	ConfigNameOrPath string
	VocabNameOrPath  string
	MergesNameOrPath string
	Device           gotch.Device
	ModelType        ModelType
	LowerCase        bool
}

// NewQuestionAnsweringConfig creates a new QuestionAnsweringConfig.
func NewQuestionAnsweringConfig(modelType ModelType, modelNameOrPath, configNameOrPath, vocabNameOrPath, mergesNameOrPath string, lowerCase bool) *QuestionAnsweringConfig {

	device := gotch.NewCuda()
	return &QuestionAnsweringConfig{
		ModelNameOrPath:  modelNameOrPath,
		ConfigNameOrPath: configNameOrPath,
		VocabNameOrPath:  vocabNameOrPath,
		MergesNameOrPath: mergesNameOrPath,
		Device:           device.CudaIfAvailable(),
		ModelType:        modelType,
		LowerCase:        lowerCase,
	}
}

// DefaultQuestionAnsweringConfig creates QuestionAnsweringConfig with default values.
func DefaultQuestionAnsweringConfig() *QuestionAnsweringConfig {
	device := gotch.NewCuda()
	modelName := "DistilBert"
	return &QuestionAnsweringConfig{
		ModelNameOrPath:  modelName,
		ConfigNameOrPath: modelName,
		VocabNameOrPath:  modelName,
		MergesNameOrPath: "",
		Device:           device.CudaIfAvailable(),
		ModelType:        DistilBert,
		LowerCase:        false,
	}
}

// QAModelOption conforms an interface with single `ForwardT` method.
type QAModelOption interface {
	ForwardT(inputIds, mask, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (startScores, endScores ts.Tensor, hiddenStates, attentions []ts.Tensor, err error)
}

type QuestionAnsweringModel struct {
	tokenizer    *tokenizer.Tokenizer
	padIdx       int
	sepIdx       int
	maxSeqLen    int
	docStride    int
	maxQueryLen  int
	maxAnswerLen int
	qaModel      pretrained.Model
	varstore     nn.VarStore
}

// NewQuesitonAnswerModel creates new corresponding QuestionAnswerModel using pretrained data from config.
func NewQuestionAnsweringModel(config *QuestionAnsweringConfig) (*QuestionAnsweringModel, error) {
	var (
		tk      *tokenizer.Tokenizer
		padId   int
		sepId   int
		qaModel pretrained.Model
		err     error
	)

	if tk, err = getTokenizer(config); err != nil {
		return nil, err
	}

	if padId, sepId, err = getPadSepIds(config); err != nil {
		return nil, err
	}

	switch reflect.TypeOf(config.ModelType).Name() {
	case "Bert":
		if qaModel, err = newBertQAModel(config); err != nil {
			return nil, err
		}
	case "Roberta":
		if qaModel, err = newRobertaQAModel(config); err != nil {
			return nil, err
		}
	default:
		return newDistilBertQAModel(config)
	}

	return &QuestionAnsweringModel{
		tokenizer:    tk,
		padIdx:       padId,
		sepIdx:       sepId,
		maxSeqLen:    384,
		docStride:    128,
		maxAnswerLen: 15,
		qaModel:      qaModel,
	}, nil
}

func newDistilBertQAModel(config *QuestionAnsweringConfig) (*QuestionAnsweringModel, error) {

	// TODO: implement
	panic("DistilBertQAModel haven't impelemented yet.")
}

func newBertQAModel(config *QuestionAnsweringConfig) (*bert.BertForQuestionAnswering, error) {
	var bertConfig *bert.BertConfig = new(bert.BertConfig)
	if err := transformer.LoadConfig(bertConfig, config.ConfigNameOrPath, nil); err != nil {
		return nil, err
	}
	var model *bert.BertForQuestionAnswering = new(bert.BertForQuestionAnswering)
	if err := transformer.LoadModel(model, config.ModelNameOrPath, bertConfig, nil, nn.NewVarStore(config.Device)); err != nil {
		return nil, err
	}

	return model, nil
}

func getTokenizer(config *QuestionAnsweringConfig) (*tokenizer.Tokenizer, error) {

	var tokModel tokenizer.Model
	if err := tokModel.(pretrained.Tokenizer).Load(config.VocabNameOrPath, config.MergesNameOrPath, nil); err != nil {
		return nil, err
	}

	tk := tokenizer.NewTokenizer(tokModel)
	return tk, nil
}

func getPadSepIds(config *QuestionAnsweringConfig) (int, int, error) {
	var (
		tk           pretrained.Tokenizer
		ok           bool
		sep          string
		padId, sepId int
		err          error
	)

	err = tk.Load(config.VocabNameOrPath, config.MergesNameOrPath, nil)
	if err != nil {
		return -1, -1, err
	}

	modelTyp := reflect.TypeOf(config.ModelType).Name()
	switch modelTyp {
	case "Bert":
		err := tk.(*bert.Tokenizer).Load(config.VocabNameOrPath, "", nil)
		if err != nil {
			return -1, -1, err
		}
		sep = "[SEP]"
		sepId, ok = tk.(*bert.Tokenizer).TokenToId(sep)
		if !ok {
			err = fmt.Errorf("Special token 'SEP' not found in vocab.\n")
			return -1, -1, err
		}
		padId = tk.(*bert.Tokenizer).GetPadding().PadId

	case "Roberta":
		sep = "</s>"
		sepId, ok = tk.(*roberta.Tokenizer).TokenToId(sep)
		if !ok {
			err = fmt.Errorf("Special token 'SEP' not found in vocab.\n")
			return -1, -1, err
		}
		padId = tk.(*roberta.Tokenizer).GetPadding().PadId

	default:
		// TODO: implement default DistilBert here
		panic("Default DistilBert has not implemented yet.")
	}

	return padId, sepId, nil
}

func newRobertaQAModel(config *QuestionAnsweringConfig) (*roberta.RobertaForQuestionAnswering, error) {
	var bertConfig *bert.BertConfig = new(bert.BertConfig)
	if err := transformer.LoadConfig(bertConfig, config.ConfigNameOrPath, nil); err != nil {
		return nil, err
	}
	var model *roberta.RobertaForQuestionAnswering = new(roberta.RobertaForQuestionAnswering)
	if err := transformer.LoadModel(model, config.ModelNameOrPath, bertConfig, nil, nn.NewVarStore(config.Device)); err != nil {
		return nil, err
	}

	return model, nil
}

// Predict performs extractive question answering given a list of `QaInputs`
func (qa *QuestionAnsweringModel) Predict(qaInputs []QAInput, topK int64, batchSize int) [][]Answer {

	var (
		examples []QAExample
		features []QAFeature
	)
	for _, input := range qaInputs {
		example := NewQAExample(input.Question, input.Context)
		examples = append(examples, *example)
	}

	for idx, example := range examples {
		feat := qa.generateFeatures(example, qa.maxSeqLen, qa.maxQueryLen, qa.docStride, idx)
		features = append(features, feat...)
	}

	var exampleTopKAnswers map[int][]Answer = make(map[int][]Answer)
	start := 0
	for start < len(features) {
		end := start + len(features) - start
		if batchSize < len(features)-start {
			end = start + batchSize
		}
		batchFeatures := features[start:end]
		var (
			inputIds, attentionsMasks []ts.Tensor
		)

		ts.NoGrad(func() {
			for _, feat := range features {
				inputTs := ts.MustOfSlice(feat.InputIds)
				inputIds = append(inputIds, inputTs)
				atnMaskTs := ts.MustOfSlice(feat.AttentionMask)
				attentionsMasks = append(attentionsMasks, atnMaskTs)
			}

			inputTsTmp := ts.MustStack(inputIds, 0)
			inputTs := inputTsTmp.MustTo(qa.varstore.Device(), true)
			attentionsTmp := ts.MustStack(attentionsMasks, 0)
			attentionsTs := attentionsTmp.MustTo(qa.varstore.Device(), true)

			var (
				startLogits, endLogits ts.Tensor
				err                    error
			)

			switch reflect.TypeOf(qa.qaModel).Name() {
			case "Bert":
				startLogits, endLogits, _, _, err = qa.qaModel.(*bert.BertForQuestionAnswering).ForwardT(inputTs, attentionsTs, ts.None, ts.None, ts.None, false)
				if err != nil {
					log.Fatal(err)
				}
			case "Roberta":
				startLogits, endLogits, _, _, err = qa.qaModel.(*roberta.RobertaForQuestionAnswering).ForwardT(inputTs, attentionsTs, ts.None, ts.None, ts.None, false)
				if err != nil {
					log.Fatal(err)
				}

			// TODO: update more models here.

			default:
				log.Fatalf("Unsupported Model type '%v'", reflect.TypeOf(qa.qaModel).Name())
			}

			startLogits.MustDetach_()
			endLogits.MustDetach_()

			var exampleIndexToFeatureEndPosition [][]int // slice of 2 elements - exampleId, maxFeatureId

			for idx, feat := range batchFeatures {
				exampleIndexToFeatureEndPosition = append(exampleIndexToFeatureEndPosition, []int{feat.ExampleIndex, idx + 1})
			}

			featureIdStart := 0
			for _, item := range exampleIndexToFeatureEndPosition {
				// item[0] is exampleId
				// item[1] is maxFeatureId
				var answers []Answer
				example := examples[item[0]]
				for featIdx := featureIdStart; featIdx < item[1]; featIdx++ {
					feature := batchFeatures[featIdx]
					start := startLogits.MustGet(featIdx)
					end := endLogits.MustGet(featIdx)
					pMask := ts.MustOfSlice(feature.PMask).MustSub1(ts.IntScalar(1), true).MustAbs(true).MustTo(start.MustDevice(), true)

					startTmpDiv := start.MustExp(false).MustSum(gotch.Float, false).MustMul(pMask, false)
					startOut := start.MustExp(false).MustDiv(startTmpDiv, true)
					start.MustDrop()

					endTmpDiv := end.MustExp(false).MustSum(gotch.Float, false).MustMul(pMask, false)
					endOut := endTmpDiv.MustExp(false).MustDiv(endTmpDiv, true)
					end.MustDrop()

					starts, ends, scores := qa.decode(startOut, endOut, topK)

					for idx := 0; idx < len(starts); idx++ {
						startPos := feature.TokenToOriginMap[int(starts[idx])]
						endPos := feature.TokenToOriginMap[int(ends[idx])]
						answer := strings.Join(example.DocTokens[startPos:endPos+1], " ")

						var start, end int

						for i, v := range example.CharToWordOffset {
							if v == startPos {
								start = i
								break
							}
						}

						// Reverse
						for i := len(example.CharToWordOffset); i > 0; i-- {
							if example.CharToWordOffset[i] == endPos {
								end = i
								break
							}
						}

						answers = append(answers, Answer{
							Score:  scores[idx],
							Start:  start,
							End:    end,
							Answer: answer,
						})
					}
				}

				featureIdStart = item[1]
				exampleAnswers := exampleTopKAnswers[item[0]]
				exampleAnswers = append(exampleAnswers, answers...)
			}

		}) // end of ts.NoGrad

		start = end
	}

	var allAnswers [][]Answer

	for exampleId := 0; exampleId < len(examples); exampleId++ {
		if answers, ok := exampleTopKAnswers[exampleId]; ok {
			answers := removeAnswerDuplicates(answers)
			allAnswers = append(allAnswers, answers)
		} else {
			allAnswers = append(allAnswers, []Answer{})
		}
	}

	return allAnswers
}

func removeAnswerDuplicates(answers []Answer) []Answer {
	var res []Answer

	for _, a := range answers {
		if !isAnswerExist(res, a) {
			res = append(res, a)
		}
	}

	return res
}

func isAnswerExist(answers []Answer, a Answer) bool {
	for _, answer := range answers {
		if reflect.DeepEqual(answer, a) {
			return true
		}
	}

	return false
}

func (qa *QuestionAnsweringModel) decode(start, end ts.Tensor, topK int64) ([]int64, []int64, []float64) {

	// TODO: delete all intermediate tensors
	outer := start.MustUnsqueeze(-1, false).MustMatmul(end.MustUnsqueeze(0, false), false)
	startDim := start.MustSize()[0]
	endDim := end.MustSize()[0]
	candidates := outer.MustTriu(0, false).MustTril(int64(qa.maxAnswerLen-1), false).MustFlatten(0, -1, false)

	var idxSort ts.Tensor
	if topK == 1 {
		idxSort = candidates.MustArgmax(0, true, false)
	} else if candidates.MustSize()[0] < topK {
		idxSort = candidates.MustArgsort(0, true, false)
	} else {
		idxSort = candidates.MustArgsort(0, true, false).MustSlice(0, 0, topK, 1, false)
	}

	var (
		startOut []int64
		endOut   []int64
		scores   []float64
	)

	for flatIndexPosition := 0; flatIndexPosition < int(idxSort.MustSize()[0]); flatIndexPosition++ {
		flatIndex := idxSort.MustInt64Value([]int64{int64(flatIndexPosition)})
		scores = append(scores, candidates.MustFloat64Value([]int64{flatIndex}))
		startOut = append(startOut, flatIndex/startDim)
		endOut = append(endOut, flatIndex%endDim)
	}

	return startOut, endOut, scores
}

func (qa *QuestionAnsweringModel) generateFeatures(qaExample QAExample, maxSeqLen, docStride, maxQueryLen, exampleIdx int) []QAFeature {

	// TODO: implement
	panic("Not implemented yet.")
}
