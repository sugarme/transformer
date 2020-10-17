package squad

import (
	"fmt"
	"log"
	// "reflect"
	"strings"

	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/tokenizer"
	// util "github.com/sugarme/tokenizer/util/slice"
	// "github.com/sugarme/transformer/pretrained"
)

var MultiSepTokensTokenizers []string = []string{
	"roberta",
	"camebert",
	"bart",
}

// Example is a single training/test example for the Squad dataset, as loaded from disk.
type Example struct {
	QAsId            string   // example unique identification
	QuestionText     string   // question string
	ContextText      string   // context string
	AnswerText       string   // answer string
	Title            string   // Title of the example
	Answers          []Answer // Default = nil. Holds answers as well as their start positions
	IsImposible      bool     // Default = false. Set to true if the example has no possible answer.
	StartPosition    int
	EndPosition      int
	DocTokens        []string // word tokens of the context.
	CharToWordOffset []int    // offset of docToken
}

// NewExample creates a Example.
//
// startPositionChar is character position of the start of the answer.
func NewExample(qasId string, question, context, answer string, startPositionChar int, title string, answers []Answer, isImpossible bool) *Example {

	start, end := 0, 0

	// Split on whitespace so that different tokens may be attributed to their original position.
	docTokens, charToWordOffset := splitContext(context)

	// Start and end positions only has a value during evaluation.
	if startPositionChar != -1 && !isImpossible {
		start = charToWordOffset[startPositionChar]
		end = charToWordOffset[startPositionChar+len(answer)-1]
		if (startPositionChar + len(answer)) > len(charToWordOffset) {
			end = charToWordOffset[len(charToWordOffset)-1]
		}
	}

	// TODO: implement
	return &Example{
		QAsId:            qasId,
		QuestionText:     question,
		ContextText:      context,
		AnswerText:       answer,
		Title:            title,
		DocTokens:        docTokens,
		CharToWordOffset: charToWordOffset,
		StartPosition:    start,
		EndPosition:      end,
	}
}

// splitContext splits input text on 'whitespace'.
// It returns tokens and their corresponding offsets.
// TODO: need a unit test
func splitContext(context string) (splits []string, offsets []int) {
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
func improveAnswerSpan(docTokens []string, inputStart, inputEnd int, tk *tokenizer.Tokenizer, originAnswerText string) (int, int) {

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

// ConvertExampleToFeatures converts input a single example to features.
// TODO: add explaination
func ConvertExampleToFeatures(tk *tokenizer.Tokenizer, example Example, maxSeqLen, docStride, maxQueryLen int, paddingStrategy, isTraining bool) []Features {

	var (
		features                   []Features
		startPosition, endPosition int
	)

	if isTraining && !example.IsImposible {
		// Get start and end position
		startPosition = example.StartPosition
		endPosition = example.EndPosition

		// If the answer cannot be found in the text, skip this example.
		actualText := strings.Join(example.DocTokens[startPosition:endPosition], " ")
		cleanedAnswerText := strings.Join(whitespaceTokenize(example.AnswerText), " ")
		if !strings.Contains(actualText, cleanedAnswerText) {
			fmt.Printf("Could not find answer: '%s' vs. '%s'\n", actualText, cleanedAnswerText)
			return []Features{}
		}
	}

	var (
		tokToOriginIndex []int
		originToTokIndex []int
		allDocTokens     []string
	)

	for i, token := range example.DocTokens {
		originToTokIndex = append(originToTokIndex, len(token))
		en, err := tk.Encode(tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(token)), false)
		if err != nil {
			log.Fatal(err)
		}
		subTokens := en.GetTokens()
		for _, subTok := range subTokens {
			tokToOriginIndex = append(tokToOriginIndex, i)
			allDocTokens = append(allDocTokens, subTok)
		}
	}

	var (
		tokStartPosition, tokEndPosition int
	)

	if isTraining && !example.IsImposible {
		tokStartPosition = originToTokIndex[example.StartPosition]
		tokEndPosition = len(allDocTokens) - 1
		if example.EndPosition < len(example.DocTokens)-1 {
			tokEndPosition = originToTokIndex[example.EndPosition+1] - 1
		}

		tokStartPosition, tokEndPosition = improveAnswerSpan(allDocTokens, tokStartPosition, tokEndPosition, tk, example.AnswerText)
	}

	// TODO: verify whether we need to modify `tokenizer.trunc` - TruncationParam
	truncParams := tokenizer.TruncationParams{
		MaxLength: maxQueryLen,
		Strategy:  tokenizer.OnlySecond,
		Stride:    docStride,
	}
	tk.WithTruncation(&truncParams)

	encoding, err := tk.EncodePair(example.QuestionText, example.ContextText, true)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("encoding: %+v\n\n", encoding)

	return features

	// TODO: implement
	// panic("has not been implemented yet.")
}

// whitespaceTokenize runs basic whitespace cleaning and splitting
// on a piece of text.
func whitespaceTokenize(text string) []string {
	stripText := strings.TrimSpace(text)
	return strings.Split(stripText, " ")
}
