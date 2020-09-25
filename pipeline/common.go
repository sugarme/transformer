package pipeline

import (
	"log"
	"reflect"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/model/wordpiece"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	"github.com/sugarme/tokenizer/processor"
	"github.com/sugarme/transformer/bert"
)

// Common blocks for generic pipelines (e.g. token classification or sequence classification)
// Provides Enums holding configuration or tokenization resources that can be used to create
// generic pipelines. The model component is defined in the generic pipeline itself as the
// pre-processing, forward pass and postprocessing differs between pipelines while basic config and
// tokenization objects don't.

// ModelType is a enum-like, identifying the type of model
type ModelType int

const (
	Bert ModelType = iota
	DistilBert
	Roberta
	XLMRoberta
	Electra
	Marian
	T5
	Albert
)

type ModelOption struct {
	model ModelType
}

type Config interface{}

// ConfigOption holds a model configuration
type ConfigOption struct {
	model  ModelType
	config Config
}

func NewBertConfigOption(config bert.BertConfig) *ConfigOption {
	return &ConfigOption{
		model:  Bert,
		config: config,
	}
}

type TokenizerType int

const (
	BertTokenizer TokenizerType = iota
	RobertaTokenizer
	XLMRobertaTokenizer
	MarianTokenizer
	T5Tokenizer
	AlbertTokenizer
)

// TokenizerOption specifies a tokenizer
type TokenizerOption struct {
	model     ModelType
	tokenizer *tokenizer.Tokenizer
}

// ConfigOption methods:
// =====================

// ConfigOptionFromFile loads configuration for corresponding model type from file.
func ConfigOptionFromFile(modelType ModelType, path string) *ConfigOption {

	var configOpt *ConfigOption

	switch reflect.TypeOf(modelType).Kind().String() {
	case "Bert":
		config := bert.ConfigFromFile(path)
		configOpt = &ConfigOption{
			model:  Bert,
			config: config,
		}

	// TODO: implement others
	// case "DistilBert":
	default:
		log.Fatalf("Invalid modelType: '%v'\n", reflect.TypeOf(modelType).Kind().String())
	}

	return configOpt
}

// GetLabelMap returns label mapping for corresponding model type.
func (co *ConfigOption) GetLabelMapping() map[int64]string {

	var labelMap map[int64]string = make(map[int64]string)

	modelTypeStr := reflect.TypeOf(co.model).Kind().String()
	switch modelTypeStr {
	case "Bert":
		labelMap = co.config.(bert.BertConfig).Id2Label

	// TODO: implement others
	default:
		log.Fatalf("ConfigOption GetLabelMapping error: invalid model type ('%v')\n", modelTypeStr)
	}

	return labelMap
}

// TOkenizerOptionFromFile loads TokenizerOption from file corresponding to model type.
func TokenizerOptionFromFile(modelType ModelType, path string) *TokenizerOption {
	modelTypeStr := reflect.TypeOf(modelType).Kind().String()

	var tk *TokenizerOption
	switch modelTypeStr {
	case "Bert":
		tk = &TokenizerOption{
			model:     modelType,
			tokenizer: getBert(path),
		}

	// TODO: implement others

	default:
		log.Fatalf("Unsupported model type: '%v'", modelTypeStr)
	}

	return tk
}

func getBert(path string) (retVal *tokenizer.Tokenizer) {
	model, err := wordpiece.NewWordPieceFromFile(path, "[UNK]")
	if err != nil {
		log.Fatal(err)
	}

	tk := tokenizer.NewTokenizer(model)

	bertNormalizer := normalizer.NewBertNormalizer(true, true, true, true)
	tk.WithNormalizer(bertNormalizer)

	bertPreTokenizer := pretokenizer.NewBertPreTokenizer()
	tk.WithPreTokenizer(bertPreTokenizer)

	var specialTokens []tokenizer.AddedToken
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("[MASK]", true))

	tk.AddSpecialTokens(specialTokens)

	sepId, ok := tk.TokenToId("[SEP]")
	if !ok {
		log.Fatalf("Cannot find ID for [SEP] token.\n")
	}
	sep := processor.PostToken{Id: sepId, Value: "[SEP]"}

	clsId, ok := tk.TokenToId("[CLS]")
	if !ok {
		log.Fatalf("Cannot find ID for [CLS] token.\n")
	}
	cls := processor.PostToken{Id: clsId, Value: "[CLS]"}

	postProcess := processor.NewBertProcessing(sep, cls)
	tk.WithPostProcessor(postProcess)

	return tk
}

// ModelType returns chosen model type
func (tk *TokenizerOption) ModelType() ModelType {
	return tk.model
}

// EncodeList encodes a slice of input string
func (tk *TokenizerOption) EncodeList(sentences []string) ([]tokenizer.Encoding, error) {
	var input []tokenizer.EncodeInput
	for _, sentence := range sentences {
		input = append(input, tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(sentence)))
	}

	return tk.tokenizer.EncodeBatch(input, true)
}

// Tokenize tokenizes input string
func (tk *TokenizerOption) Tokenize(sentence string) ([]string, error) {

	input := tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(sentence))

	encoding, err := tk.tokenizer.Encode(input, true)
	if err != nil {
		return nil, err
	}

	return encoding.Tokens, nil
}

// AddSpecialTokens adds special tokens to tokenizer
func (tk *TokenizerOption) AddSpecialTokens(tokens []string) {

	var addedToks []tokenizer.AddedToken
	for _, tok := range tokens {
		addedToks = append(addedToks, tokenizer.NewAddedToken(tok, true))
	}

	tk.tokenizer.AddSpecialTokens(addedToks)
}

// TokensToIds converts a slice of tokens to corresponding Ids.
func (tk *TokenizerOption) TokensToIds(tokens []string) (ids []int64, ok bool) {
	for _, tok := range tokens {
		id, ok := tk.tokenizer.TokenToId(tok)
		if !ok {
			return nil, false
		}
		ids = append(ids, int64(id))
	}

	return ids, true
}

// PadId returns a PAD id if any.
func (tk *TokenizerOption) PadId() (id int64, ok bool) {
	paddingParam := tk.tokenizer.GetPadding()
	if paddingParam == nil {
		return -1, false
	}

	return int64(paddingParam.PadId), true
}

// SepId returns a SEP id if any.
// If optional sepOpt is not specify, default value is "[SEP]"
func (tk *TokenizerOption) SepId(sepOpt ...string) (id int64, ok bool) {

	sep := "[SEP]" // default sep string
	if len(sepOpt) > 0 {
		sep = sepOpt[0]
	}

	i, ok := tk.tokenizer.TokenToId(sep)
	return int64(i), ok
}
