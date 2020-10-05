package pipeline

import (
	"fmt"
	"reflect"

	"github.com/sugarme/gotch"
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
	InputIds         []int
	AttentionMask    []int
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
	// varstore     nn.VarStore
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
	if err := transformer.LoadModel(model, config.ModelNameOrPath, bertConfig, nil, config.Device); err != nil {
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
	if err := transformer.LoadModel(model, config.ModelNameOrPath, bertConfig, nil, config.Device); err != nil {
		return nil, err
	}

	return model, nil
}
