package squad

import (
	ts "github.com/sugarme/gotch/tensor"
)

var MultiSepTokensTokenizers []string = []string{
	"roberta",
	"camebert",
	"bart",
}

// Example is a single training/test example for the Squad dataset, as loaded from disk.
type Example struct {
	QAsId             string   // example unique identification
	QuestionText      string   // question string
	ContextText       string   // context string
	AnswerText        string   // answer string
	StartPositionChar int      // Character position of the start of the answer
	Title             string   // Title of the example
	Answers           []Answer // Default = nil. Holds answers as well as their start positions
	IsImposible       bool     // Default = false. Set to true if the example has no possible answer.
}

func NewExample(qasId int, question, context, answer string, startPositionChar int, title string, answers []Answer, isImpossible bool) *Example {

	// TODO: implement
	return &Example{}
}

// Answer is Squad answer struct.
type Answer struct {
	// TODO: add fields.
}

// Features are SINGLE squad example features to be fed to the model.
//
// Those features are model-specific and can be crafted using method `SquadExample.ConvertExampleToFeatures`.
type Features struct {
	QAsId             string // example unique identification
	InputIds          []int  // Indices of input sequence tokens in the vocabulary.
	AttentionMask     string // Mask to avoid performing attention on padding token indices.
	TokenTypeIds      []int  // Segment token indices to indicate first and second portions of the input.
	ClsIndex          int    // Index of the CLS token.
	PMask             int    // Mask identifying tokens that can be answers versus tokens that cannot (1 not in the answer, 0 in the answer).
	ExampleIndex      int    // Index of the example.
	UniqueId          int    // The unique feature identifier.
	ParagraphLen      int    // The length of the context.
	TokenIsMaxContext []bool // A bool slice identifying which tokens have their maximum context in this feature.
	// NOTE. If a token does not have their maximum context in this feature, it means that another feature has more
	// information related to that token and should be prioritized over this feature for that token.
	Tokens         []string    // Slice of tokens corresponding to the input Ids.
	TokenToOrigMap map[int]int // Mapping between the tokens and the original text needed in order to identify the answer.
	StartPosition  int         // Start of the answer token index.
	EndPosition    int         // End of the answer token index.
}

// NewFeature creates new Squad Features.
func NewFeature(inputIds []int, attentionMask string, tokenTypeIds []int, clsIndex int, pMask int, exampleIndex int, uniqueId int, paragraphLen int, tokenIsMaxContext []bool, tokens []string, tokenToOrigMap map[int]int, startPosition, endPosition int, isImposible bool, qasId string) *Features {
	return &Features{
		InputIds:          inputIds,
		AttentionMask:     attentionMask,
		TokenTypeIds:      tokenTypeIds,
		ClsIndex:          clsIndex,
		PMask:             pMask,
		ExampleIndex:      exampleIndex,
		UniqueId:          uniqueId,
		ParagraphLen:      paragraphLen,
		TokenIsMaxContext: tokenIsMaxContext,
		Tokens:            tokens,
		TokenToOrigMap:    tokenToOrigMap,
		StartPosition:     startPosition,
		EndPosition:       endPosition,
		QAsId:             "",
	}
}

// Result constructs a Squad result that can be used to evaluate a model's output on the SQuAD dataset.
type Result struct {
	UniqueId       string    // The unique identifier corresponding to that example.
	StartLogits    ts.Tensor // The logits corresponding to the start of the answer.
	EndLogits      ts.Tensor // The logits corresponding to the end of the answer.
	StartStopIndex int
	EndStopIndex   int
	ClsLogits      ts.Tensor
}

func NewResult(uniqueId string, start, end ts.Tensor, startStopIndex, endStopIndex int, clsLogits ts.Tensor) *Result {
	return &Result{
		UniqueId:       uniqueId,
		StartLogits:    start,
		EndLogits:      end,
		StartStopIndex: -1,
		EndStopIndex:   -1,
		ClsLogits:      ts.None,
	}
}

// improveAnswerSpan returns answer spans that better match the annotated answer.
func improveAnswerSpan(docTokens []string, inputStart, inputEnd ts.Tensor, originAnswerText string) (ts.Tensor, ts.Tensor) {

	// TODO: implement
	panic("has not been implemented yet.")
}

// isMaxContext checks whether this is the 'max context' doc span for the token.
func isMaxContext(docSpans []int, currSpanIdx, position int) bool {

	// TODO: implement
	panic("has not been implemented yet.")
}

// newIsMaxContext checks whether this is the 'max context' doc span for the token.
func newIsMaxContext(docSpans []int, currSpanIdx, position int) bool {

	// TODO: implement
	panic("has not been implemented yet.")
}

func isWhiteSpace(char rune) bool {
	if char == ' ' || char == '\t' || char == '\r' || char == '\n' || char == 0x202F {
		return true
	}
	return false
}

func ConvertExampleToFeatures(example Example, maxSeqLen int, docStride, maxQueryLen, paddingStrategy, isTraining bool) []Features {

	// var features []Features
	if isTraining {
		// Get start and end position
		// startPosition := example.
	}

	// TODO: implement
	panic("has not been implemented yet.")
}
