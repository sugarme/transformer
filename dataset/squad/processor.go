package squad

import (
	"fmt"
	"log"
	"strings"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/tensor"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/tokenizer"
	util "github.com/sugarme/tokenizer/util/slice"
	// "github.com/sugarme/transformer/pretrained"
)

var MultiSepTokensTokenizers []string = []string{
	"roberta",
	"camembert",
	"bart",
}

// Example is a single training/test example for the Squad dataset, as loaded from disk.
type Example struct {
	QAsId         string   // example unique identification
	QuestionText  string   // question string
	ContextText   string   // context string
	AnswerText    string   // answer string
	Title         string   // Title of the example
	Answers       []Answer // Default = nil. Holds answers as well as their start positions
	IsImposible   bool     // Default = false. Set to true if the example has no possible answer.
	StartPosition int      // index of start rune ("character") of the answer
}

// NewExample creates a Example.
//
// Params:
// - qasId: unique Id of QAs in SQuAD dataset
// - context: paragraph text
// - answer: answer text
// - startPositionChar:  first rune position of the answer string
// - title: title of the document (article) in SQuAD dataset
// - isImpossibleOpt: optional param set whether the example has no possible answer. Default=False
func NewExample(qasId string, question, context, answer string, startPositionChar int, title string, isImpossibleOpt ...bool) *Example {

	isImpossible := false
	if len(isImpossibleOpt) > 0 {
		isImpossible = isImpossibleOpt[0]
	}

	if isImpossible {
		startPositionChar = -1
	}

	return &Example{
		QAsId:         qasId,
		QuestionText:  question,
		ContextText:   context,
		AnswerText:    answer,
		Title:         title,
		StartPosition: startPositionChar,
	}
}

// Feature are SINGLE squad example feature to be fed to the model.
//
// This feature is model-specific and can be crafted using method `SquadExample.ConvertExampleToFeatures`.
type Feature struct {
	QAsId             string // example unique identification
	InputIds          []int  // Indices of input sequence tokens in the vocabulary.
	AttentionMask     []int  // Mask to avoid performing attention on padding token indices.
	TokenTypeIds      []int  // Segment token indices to indicate first and second portions of the input.
	ClsIndex          int    // Index of the CLS token.
	PMask             []int  // Mask identifying tokens that can be answers versus tokens that cannot (1 not in the answer, 0 in the answer).
	ExampleIndex      int    // Index of the example.
	UniqueId          int    // The unique feature identifier.
	TokenIsMaxContext []bool // A bool slice identifying which tokens have their maximum context in this feature.
	// NOTE. If a token does not have their maximum context in this feature, it means that another feature has more
	// information related to that token and should be prioritized over this feature for that token.
	Tokens        []string // Slice of tokens corresponding to the input Ids.
	StartPosition int      // Index of the first answer token. Value=0 if there's no answer
	EndPosition   int      // Index of the last answer token. Value=0 if there's no answer
	IsImposible   bool     // whether feature has no possible answer. False means feature has no possible answer.
}

// NewFeature creates new Squad Features.
func NewFeature(inputIds []int, attentionMask []int, tokenTypeIds []int, clsIndex int, pMask []int, exampleIndex int, uniqueId int, tokenIsMaxContext []bool, tokens []string, startPosition, endPosition int, isImposible bool, qasId string) *Feature {
	return &Feature{
		InputIds:          inputIds,
		AttentionMask:     attentionMask,
		TokenTypeIds:      tokenTypeIds,
		ClsIndex:          clsIndex,
		PMask:             pMask,
		ExampleIndex:      exampleIndex,
		UniqueId:          uniqueId,
		TokenIsMaxContext: tokenIsMaxContext,
		Tokens:            tokens,
		StartPosition:     startPosition,
		EndPosition:       endPosition,
		QAsId:             qasId,
		IsImposible:       isImposible,
	}
}

// Result constructs a Squad result that can be used to evaluate a model's output on the SQuAD dataset.
type Result struct {
	UniqueId       string     // The unique identifier corresponding to that example.
	StartLogits    *ts.Tensor // The logits corresponding to the start of the answer.
	EndLogits      *ts.Tensor // The logits corresponding to the end of the answer.
	StartStopIndex int
	EndStopIndex   int
	ClsLogits      *ts.Tensor
}

func NewResult(uniqueId string, start, end *ts.Tensor, startStopIndex, endStopIndex int, clsLogits *ts.Tensor) *Result {
	return &Result{
		UniqueId:       uniqueId,
		StartLogits:    start,
		EndLogits:      end,
		StartStopIndex: -1,
		EndStopIndex:   -1,
		ClsLogits:      ts.NewTensor(),
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
//
// Params:
// - tk: tokenizer to use
// - sepToken: separator token
// - clsIndex: index of the cls token
// - answerStart: start offset of answer on context sequence.
// - answerEnd: end offset of answer on context sequence.
func ConvertExampleToFeatures(tk *tokenizer.Tokenizer, sepToken string, clsIndex int, example Example, answerStart, answerEnd int) []Feature {

	var (
		features []Feature
	)

	// NOTE. tokenizer `TruncationParams` and `PaddingParams` should be configued up-stream
	encoding, err := tk.EncodePair(example.QuestionText, example.ContextText, true)
	if err != nil {
		fmt.Println("EncodePair error:")
		log.Fatal(err)
	}

	var (
		spans []tokenizer.Encoding
	)

	// 1. Encoding
	selfSpan := encoding
	selfSpan.Overflowing = []tokenizer.Encoding{}
	spans = append(spans, *selfSpan)

	// 2. Overflowing if any
	spans = append(spans, encoding.Overflowing...)

	for spanIndex, span := range spans {
		// 3. pMask
		var pMask []int
		var sepIdx int
		var maxContext []bool
		for i, tok := range span.Tokens {
			if tok == sepToken {
				sepIdx = i
				break
			}
		}
		for i := 0; i < len(span.Tokens); i++ {
			// Mask question tokens
			if i <= sepIdx {
				pMask = append(pMask, 1)
			} else {
				pMask = append(pMask, 0)
			}

			// determine whether token is "Max Context"
			isMaxCtx := isMaxContext(spans, spanIndex, span.Words[i])
			maxContext = append(maxContext, isMaxCtx)
		}

		// 4. Find token indices of answer in encoded tokens
		var startA int = -1                   // index of first answer token
		var endA int = -1                     // index of last answer token
		for i, offset := range span.Offsets { // i corresponds to token index
			if span.TypeIds[i] != 1 { // question part
				continue
			}
			if answerStart >= offset[0] && answerStart <= offset[1] {
				startA = i
				break
			}
		}

		// If reconstruct answer text does matched with the original, skip this example.
		// NOTE. This would exclude any overflowings that don't contain the answer sequence.
		if startA != -1 && string([]rune(example.AnswerText)[0]) == string([]rune(span.Tokens[startA])[0]) {
			queryTk := tk
			queryTk.WithPadding(nil)

			aEncoding, err := queryTk.EncodeSingle(example.AnswerText, false)
			if err != nil {
				log.Fatal(err)
			}
			endA = len(aEncoding.Tokens) + startA
			// fmt.Printf("origin: %q - decode: %q\n", example.AnswerText, tk.Decode(encoding.Ids[startA:endA], true))
			// fmt.Printf("%q\n", tk.Decode(span.Ids, true))

		} else {
			fmt.Printf("##### Answer not matched. Skip this.#####\n")
			continue
		}

		feature := NewFeature(span.Ids, span.AttentionMask, span.TypeIds, clsIndex, pMask, 0, 0, maxContext, span.Tokens, startA, endA, example.IsImposible, example.QAsId)

		// Add exampleIndex
		feature.ExampleIndex = spanIndex
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
// - tkName: name of tokenizer (whether it uses multiple sep tokens: ["roberta", "bart", "camebert"]).
// - maxSeqLen: maximal length of the input sequence to feed into the model (count in number of tokens)
// - maxQueryLen: maximal length of the question (count in number of tokens)
// - docStride: the stride (step size) tokenizer will use to split the encoding overflowing if it occurs.
// 	 E.g. total overflowing tokens=20, maxSeqLen=10, docStride=5, there will be 4 encodings of 10 tokens.
// - sepToken: sep token used in tokenizer (e.g. BERT tokenizer uses "[SEP]")
// - padToken: pad token used in tokenizer (e.g. BERT tokenizer uses "[PAD]")
// - clsIndex: index position of the cls token in encoded input after tokenizing (e.g. BERT tokenizer [CLS] index = 0)
// - isTraining: whether to config features for training (added Answer)
// - returnTensorDataset: whether to stack Feature fields to a tensor.
//
// NOTE. returning tensor depends on input params:
// - returnTensor=false: return tensor.None
// - isTraining=false: return tensor of size [6, numOfFeatures, maxSeqLen]:
//   + inputIds
//   + attentionMasks
//   + tokenTypeIds
//   + featureIndexes
//   + clsIndex (repeated values to make size=maxSeqLen)
//   + pMasks
// - isTraining=true: return tensor of size = [8, numOfFeatures, maxSeqLen]
//   + inputIds
//   + attentionMasks
//   + tokenTypeIds
//   + startPosition (repeated values to make size=maxSeqLen)
//   + endPosition (repeated values to make size=maxSeqLen)
//   + clsIndex (repeated values to make size=maxSeqLen)
//   + pMasks
//   + isImpossible (repeated values to make size=maxSeqLen)
func ConvertExamplesToFeatures(examples []Example, tk *tokenizer.Tokenizer, tkName string, maxSeqLen, docStride, maxQueryLen int, sepToken, padToken string, clsIndex int, isTraining bool, returnTensorDataset bool) ([]Feature, *ts.Tensor) {
	// TODO: setup running in parallel
	var features []Feature

	for _, example := range examples {
		// verify that context contains the answer. If not, skip this example for training.
		if isTraining && example.IsImposible {
			continue
		}

		// if context does not contain answer, skip it
		if !strings.Contains(example.ContextText, example.AnswerText) {
			continue
		}

		// Determine answer start/end (byte) offsets
		oStart := len(string([]rune(example.ContextText))[0:example.StartPosition])
		oEnd := oStart + len(example.AnswerText)

		// 1. Truncate question if needed
		queryTruncParams := tokenizer.TruncationParams{
			MaxLength: maxQueryLen,
			Strategy:  tokenizer.OnlyFirst,
			Stride:    0,
		}
		queryTk := tk
		queryTk.WithTruncation(&queryTruncParams)

		queryPaddingStrategy := tokenizer.NewPaddingStrategy(tokenizer.WithFixed(0)) // fixed length
		queryPadId, ok := tk.TokenToId(padToken)
		if !ok {
			log.Fatalf("'ConvertExampleToFeatures' method call error: cannot find pad token in the vocab.\n")
		}

		queryPaddingParams := tokenizer.PaddingParams{
			Strategy:  *queryPaddingStrategy,
			Direction: tokenizer.Right, // padding right
			PadId:     queryPadId,
			PadTypeId: 1,
			PadToken:  padToken,
		}
		queryTk.WithPadding(&queryPaddingParams)

		queryEncoding, err := queryTk.EncodeSingle(example.QuestionText, false)
		if err != nil {
			fmt.Printf("Error at EncodeSingle \n")
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
		// <s> Question </s> </s> ContextSequence </s>
		var numAddedTokens int = 3
		if util.Contain(tkName, MultiSepTokensTokenizers) {
			numAddedTokens += 1
		}

		// Step on context only.
		stride := docStride + len(queryEncoding.Tokens) + numAddedTokens

		// Setup truncation Params
		truncParams := tokenizer.TruncationParams{
			MaxLength: maxSeqLen,
			Strategy:  tokenizer.OnlySecond, // truncate the context only
			Stride:    stride,
		}
		contextTk.WithTruncation(&truncParams)

		// Setup padding Params

		paddingStrategy := tokenizer.NewPaddingStrategy(tokenizer.WithFixed(maxSeqLen)) // fixed length
		padId, ok := tk.TokenToId(padToken)
		if !ok {
			log.Fatalf("'ConvertExampleToFeatures' method call error: cannot find pad token in the vocab.\n")
		}

		paddingParams := tokenizer.PaddingParams{
			Strategy:  *paddingStrategy,
			Direction: tokenizer.Right, // padding right
			PadId:     padId,
			PadTypeId: 1,
			PadToken:  padToken,
		}
		contextTk.WithPadding(&paddingParams)

		exFeatures := ConvertExampleToFeatures(contextTk, sepToken, clsIndex, example, oStart, oEnd)
		features = append(features, exFeatures...)
	}

	// Add feature uniqueId
	uniqueId := 1000000000
	for i := 0; i < len(features); i++ {
		features[i].UniqueId = uniqueId
		uniqueId++
	}

	if returnTensorDataset {
		var (
			allInputIds       [][]int64
			allAttentionMasks [][]int64
			allTokenTypeIds   [][]int64
			allPMask          [][]int64
			allStartPositions []int64
			allEndPositions   []int64
			allClsIndex       []int64
			allIsImpossible   []bool

			dataset *ts.Tensor
		)

		for _, feat := range features {
			allInputIds = append(allInputIds, toInt64(feat.InputIds))
			allAttentionMasks = append(allAttentionMasks, toInt64(feat.AttentionMask))
			allTokenTypeIds = append(allTokenTypeIds, toInt64(feat.TokenTypeIds))
			allPMask = append(allPMask, toInt64(feat.PMask))
			allStartPositions = append(allStartPositions, int64(feat.StartPosition))
			allEndPositions = append(allEndPositions, int64(feat.EndPosition))
			allClsIndex = append(allClsIndex, int64(feat.ClsIndex))
			allIsImpossible = append(allIsImpossible, feat.IsImposible)
		}

		featureNum := int64(len(features))
		shape := []int64{featureNum, int64(maxSeqLen)}
		inputIds, err := ts.NewTensorFromData(allInputIds, shape)
		if err != nil {
			log.Fatalf("ConvertExamplesToFeatures Method - Convert InputIds error: '%v'\n", err)
		}
		inputIdsTs := inputIds.MustUnsqueeze(0, true)
		attentionMasks, err := ts.NewTensorFromData(allAttentionMasks, shape)
		if err != nil {
			log.Fatalf("ConvertExamplesToFeatures Method - Convert AttentionMasks error: '%v'\n", err)
		}
		attentionMasksTs := attentionMasks.MustUnsqueeze(0, true)

		tokenTypeIds, err := ts.NewTensorFromData(allTokenTypeIds, shape)
		if err != nil {
			log.Fatalf("ConvertExamplesToFeatures Method - Convert TokenTypeIds error: '%v'\n", err)
		}
		tokenTypeIdsTs := tokenTypeIds.MustUnsqueeze(0, true)

		pMasks, err := ts.NewTensorFromData(allPMask, shape)
		if err != nil {
			log.Fatalf("ConvertExamplesToFeatures Method - Convert PMasks error: '%v'\n", err)
		}
		pMasksTs := pMasks.MustUnsqueeze(0, true)

		clsIndexes, err := ts.NewTensorFromData(allClsIndex, []int64{featureNum})
		if err != nil {
			log.Fatalf("ConvertExamplesToFeatures Method - Convert ClsIndexes error: '%v'\n", err)
		}
		clsIndexesTs := clsIndexes.MustUnsqueeze(1, true).MustExpand([]int64{-1, int64(maxSeqLen)}, true, true).MustUnsqueeze(0, true)

		var featTensors []ts.Tensor
		if isTraining {

			startPositions, err := ts.NewTensorFromData(allStartPositions, []int64{featureNum})
			if err != nil {
				log.Fatalf("ConvertExamplesToFeatures Method - Convert StartPositions error: '%v'\n", err)
			}
			startPositionsTs := startPositions.MustUnsqueeze(1, true).MustExpand([]int64{-1, int64(maxSeqLen)}, true, true).MustUnsqueeze(0, true)

			endPositions, err := ts.NewTensorFromData(allEndPositions, []int64{featureNum})
			if err != nil {
				log.Fatalf("ConvertExamplesToFeatures Method - Convert EndPositions error: '%v'\n", err)
			}
			endPositionsTs := endPositions.MustUnsqueeze(1, true).MustExpand([]int64{-1, int64(maxSeqLen)}, true, true).MustUnsqueeze(0, true)
			isImposibles := ts.MustOfSlice(allIsImpossible).MustView([]int64{featureNum}, true)
			if err != nil {
				log.Fatalf("ConvertExamplesToFeatures Method - Convert IsImposibles error: '%v'\n", err)
			}
			isImposiblesTs := isImposibles.MustUnsqueeze(1, true).MustExpand([]int64{-1, int64(maxSeqLen)}, true, true).MustUnsqueeze(0, true).MustTotype(gotch.Int64, true)

			defer startPositionsTs.MustDrop()
			defer endPositionsTs.MustDrop()
			defer isImposiblesTs.MustDrop()

			featTensors = []ts.Tensor{*inputIdsTs, *attentionMasksTs, *tokenTypeIdsTs, *startPositionsTs, *endPositionsTs, *clsIndexesTs, *pMasksTs, *isImposiblesTs}
		} else {
			allFeatureIndexTs := ts.MustArange(tensor.IntScalar(int64(len(features))), gotch.Int64, gotch.CPU).MustUnsqueeze(1, true).MustExpand([]int64{-1, int64(maxSeqLen)}, true, true).MustUnsqueeze(0, true)

			featTensors = []ts.Tensor{*inputIdsTs, *attentionMasksTs, *tokenTypeIdsTs, *allFeatureIndexTs, *clsIndexesTs, *pMasksTs}
			defer allFeatureIndexTs.MustDrop()
		}

		dataset = ts.MustCat(featTensors, 0)

		inputIdsTs.MustDrop()
		attentionMasksTs.MustDrop()
		tokenTypeIdsTs.MustDrop()
		pMasksTs.MustDrop()
		clsIndexesTs.MustDrop()

		return features, dataset
	}

	return features, ts.None
}

func toInt64(data []int) []int64 {
	var data64 []int64
	for _, v := range data {
		data64 = append(data64, int64(v))
	}

	return data64
}
