package squad

import (
	"fmt"
	"log"
	"strings"

	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/tokenizer"
	util "github.com/sugarme/tokenizer/util/slice"
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
	AttentionMask     []int  // Mask to avoid performing attention on padding token indices.
	TokenTypeIds      []int  // Segment token indices to indicate first and second portions of the input.
	ClsIndex          int    // Index of the CLS token.
	PMask             []int  // Mask identifying tokens that can be answers versus tokens that cannot (1 not in the answer, 0 in the answer).
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
func NewFeature(inputIds []int, attentionMask []int, tokenTypeIds []int, clsIndex int, pMask []int, exampleIndex int, uniqueId int, paragraphLen int, tokenIsMaxContext []bool, tokens []string, startPosition, endPosition int, isImposible bool, qasId string) *Features {
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
		// TokenToOrigMap:    tokenToOrigMap,
		StartPosition: startPosition,
		EndPosition:   endPosition,
		QAsId:         "",
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

// ConvertExampleToFeatures converts a single example to features.
func ConvertExampleToFeatures(tk *tokenizer.Tokenizer, example Example, sepToken string, clsIndex int, isTraining bool) []Features {

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

	// NOTE. tokenizer `TruncationParams` and `PaddingParams` should be configued up-stream
	encoding, err := tk.EncodePair(example.QuestionText, example.ContextText, true)
	if err != nil {
		log.Fatal(err)
	}

	var spans []tokenizer.Encoding
	// 1. Encoding
	selfSpan := encoding
	selfSpan.Overflowing = []tokenizer.Encoding{}
	spans = append(spans, *selfSpan)
	// 2. Overflowing if any
	spans = append(spans, encoding.Overflowing...)

	for spanIndex, span := range spans {

		// 2. pMask
		var pMask []int
		var sepIdx int
		var maxContext []bool
		for i, tok := range span.Tokens {
			if tok == sepToken {
				sepIdx = i
			}
		}
		for i := 0; i < len(span.Tokens); i++ {
			// Mask question tokens
			if i <= sepIdx {
				pMask = append(pMask, 1)
			}
			pMask = append(pMask, 0)

			// determine whether token is "Max Context"
			isMaxCtx := isMaxContext(spans, spanIndex, span.Words[i])
			maxContext = append(maxContext, isMaxCtx)
		}

		feature := NewFeature(span.Ids, span.AttentionMask, span.TypeIds, clsIndex, pMask, 0, 0, len(encoding.Tokens), maxContext, span.Tokens, span.Words[0], span.Words[len(span.Tokens)-1], example.IsImposible, example.QAsId)

		features = append(features, *feature)
	}

	return features
}

// whitespaceTokenize runs basic whitespace cleaning and splitting
// on a piece of text.
func whitespaceTokenize(text string) []string {
	stripText := strings.TrimSpace(text)
	return strings.Split(stripText, " ")
}

// isMaxContext checks whether the given current doc span index is a "max context" for the token.
func isMaxContext(docSpans []tokenizer.Encoding, currentSpanIndex, position int) bool {
	var bestScore float64 = 0
	var bestSpanIndex int = -1
	for spanIndex, docSpan := range docSpans {
		if position < docSpan.Words[0] || position > docSpan.Words[len(docSpan.Words)-1] {
			continue
		}
		// For example: position is 7
		// 4 5 6 [7] 8 9 10 11
		// <- L=3-|-- R=4---->
		numLeftContext := position - docSpan.Words[0]
		numRightContext := len(docSpan.Words) - position
		score := float64(numLeftContext) + 0.01*float64(len(docSpan.Words))
		if numLeftContext > numRightContext {
			score = float64(numRightContext) + 0.01*float64(len(docSpan.Words))
		}

		if bestScore == -1 || score > bestScore {
			bestScore = score
			bestSpanIndex = spanIndex
		}
	}

	return bestSpanIndex == currentSpanIndex
}

// ConvertExamplesToFeatures converts a list of examples into a list of features that can be
// directly fed into a model. It is model-dependant and takes advantage of many of the tokenizer's
// features to create the model's inputs.
//
// Params:
// - examples: Slice of Examples to convert
// - tk: corresponding Tokenizer to be used with the model
// - tkName: name of tokenizer (whether it uses multiple sep tokens)
// - maxSeqLen: maximal length of the input sequence to feed into the model (count in number of tokens)
// - maxQueryLen: maximal length of the question (count in number of tokens)
// - docStride: the stride (step size) tokenizer will use to split the encoding overflowing if it occurs.
// 	 E.g. total overflowing tokens=20, maxSeqLen=10, docStride=5, there will be 4 encodings of 10 tokens.
// - sepToken: sep token used in tokenizer (e.g. BERT tokenizer uses "[SEP]")
// - clsIndex: index position of the cls token in encoded input after tokenizing (e.g. BERT tokenizer [CLS] index = 0)
func ConvertExamplesToFeatures(examples []Example, tk *tokenizer.Tokenizer, tkName string, maxSeqLen, docStride, maxQueryLen int, sepToken string, clsIndex int, isTraining bool) [][]Features {
	// TODO: do converting in parallel
	var featuresSet [][]Features

	for _, example := range examples {
		// 1. Truncate question if needed
		queryTruncParams := tokenizer.TruncationParams{
			MaxLength: maxQueryLen,
			Strategy:  tokenizer.OnlyFirst,
			Stride:    0,
		}
		queryTk := tk
		queryTk.WithTruncation(&queryTruncParams)
		queryEncoding, err := queryTk.EncodeSingle(example.QuestionText, false)
		if err != nil {
			log.Fatal(err)
		}
		truncatedQuery := queryTk.Decode(queryEncoding.Ids, true)
		example.QuestionText = truncatedQuery

		// 2. Convert example
		contextTk := tk

		// NOTE. number of added tokens will depend on specific tokenizer model.
		// E.g. BERT tokenizer encodes a pair(question, context) will add 3 special tokens
		// [CLS] QuestionSequence [SEP] ContextSequence [SEP]
		// RoBERTa uses 2 sep tokens instead of one.
		var numAddedTokens int = 2
		if util.Contain(tkName, MultiSepTokensTokenizers) {
			numAddedTokens += 1
		}

		// stride will add in the length of question and number of added tokens.
		stride := docStride + len(queryEncoding.Tokens) + numAddedTokens

		// Setup truncation Params
		truncParams := tokenizer.TruncationParams{
			MaxLength: maxSeqLen,
			Strategy:  tokenizer.OnlySecond, // truncate the context only
			Stride:    stride,
		}
		contextTk.WithTruncation(&truncParams)

		// Setup padding Params
		padding := tk.GetPadding()
		paddingStrategy := tokenizer.NewPaddingStrategy() // defaultVal = "BatchLongest"

		paddingParams := tokenizer.PaddingParams{
			Strategy:  *paddingStrategy,
			Direction: tokenizer.Right, // padding right
			PadId:     padding.PadId,
			PadTypeId: padding.PadTypeId,
			PadToken:  padding.PadToken,
		}
		contextTk.WithPadding(&paddingParams)

		features := ConvertExampleToFeatures(contextTk, example, sepToken, clsIndex, isTraining)
		featuresSet = append(featuresSet, features)
	}

	return featuresSet
}
