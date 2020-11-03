package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/model/wordpiece"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	"github.com/sugarme/tokenizer/processor"
	"github.com/sugarme/transformer/bert"
)

func main() {
	bertForMaskedLM()
	// bertForSequenceClassification()
}

func getBert() (retVal *tokenizer.Tokenizer) {

	vocabFile := "../../data/bert/vocab.txt"

	model, err := wordpiece.NewWordPieceFromFile(vocabFile, "[UNK]")
	if err != nil {
		log.Fatal(err)
	}

	tk := tokenizer.NewTokenizer(model)
	fmt.Printf("Vocab size: %v\n", tk.GetVocabSize(false))

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

func bertForMaskedLM() {

	device := gotch.CPU
	vs := nn.NewVarStore(device)

	config, _ := bert.ConfigFromFile("../../data/bert/config.json")
	// fmt.Printf("Bert Configuration:\n%+v\n", config)

	model := bert.NewBertForMaskedLM(vs.Root(), config)
	err := vs.Load("../../data/bert/model.ot")
	if err != nil {
		log.Fatalf("Load model weight error: \n%v", err)
	}

	// fmt.Printf("Varstore weights have been loaded\n")
	// fmt.Printf("Num of variables: %v\n", len(vs.Variables()))
	// fmt.Printf("%v\n", vs.Variables())
	// fmt.Printf("Bert is Decoder: %v\n", model.bert.IsDecoder)

	tk := getBert()
	sentence1 := "Looks like one [MASK] is missing"
	sentence2 := "It was a very nice and [MASK] day"

	var input []tokenizer.EncodeInput
	input = append(input, tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(sentence1)))
	input = append(input, tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(sentence2)))

	encodings, err := tk.EncodeBatch(input, true)
	if err != nil {
		log.Fatal(err)
	}

	// Find max length of token Ids from slice of encodings
	var maxLen int = 0
	for _, en := range encodings {
		if len(en.Ids) > maxLen {
			maxLen = len(en.Ids)
		}
	}

	var tensors []ts.Tensor
	for _, en := range encodings {
		var tokInput []int64 = make([]int64, maxLen)
		for i := 0; i < len(en.Ids); i++ {
			tokInput[i] = int64(en.Ids[i])
		}

		tensors = append(tensors, *ts.TensorFrom(tokInput))
	}

	inputTensor := ts.MustStack(tensors, 0).MustTo(device, true)
	// inputTensor.Print()

	var output *ts.Tensor
	ts.NoGrad(func() {
		output, _, _ = model.ForwardT(inputTensor, ts.None, ts.None, ts.None, ts.None, ts.None, ts.None, false)
	})

	index1 := output.MustGet(0).MustGet(4).MustArgmax([]int64{0}, false, false).Int64Values()[0]
	index2 := output.MustGet(1).MustGet(7).MustArgmax([]int64{0}, false, false).Int64Values()[0]

	word1, ok := tk.IdToToken(int(index1))
	if !ok {
		fmt.Printf("Cannot find a corresponding word for the given id (%v) in vocab.\n", index1)
	}
	fmt.Printf("Input: '%v' \t- Output: '%v'\n", sentence1, word1)

	word2, ok := tk.IdToToken(int(index2))
	if !ok {
		fmt.Printf("Cannot find a corresponding word for the given id (%v) in vocab.\n", index2)
	}
	fmt.Printf("Input: '%v' \t- Output: '%v'\n", sentence2, word2)
}

func bertForSequenceClassification() {
	device := gotch.CPU
	vs := nn.NewVarStore(device)

	config, _ := bert.ConfigFromFile("../../data/bert/config.json")

	var dummyLabelMap map[int64]string = make(map[int64]string)
	dummyLabelMap[0] = "positive"
	dummyLabelMap[1] = "negative"
	dummyLabelMap[3] = "neutral"

	config.Id2Label = dummyLabelMap
	config.OutputAttentions = true
	config.OutputHiddenStates = true
	model := bert.NewBertForSequenceClassification(vs.Root(), config)
	tk := getBert()

	// Define input
	sentence1 := "Looks like one thing is missing"
	sentence2 := `It's like comparing oranges to apples`
	var input []tokenizer.EncodeInput
	input = append(input, tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(sentence1)))
	input = append(input, tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(sentence2)))
	encodings, err := tk.EncodeBatch(input, true)
	if err != nil {
		log.Fatal(err)
	}

	// Find max length of token Ids from slice of encodings
	var maxLen int = 0
	for _, en := range encodings {
		if len(en.Ids) > maxLen {
			maxLen = len(en.Ids)
		}
	}

	fmt.Printf("encodings: %v\n", encodings)
	var tensors []ts.Tensor
	for _, en := range encodings {
		var tokInput []int64 = make([]int64, maxLen)
		for i := 0; i < len(en.Ids); i++ {
			tokInput[i] = int64(en.Ids[i])
		}

		tensors = append(tensors, *ts.TensorFrom(tokInput))
	}

	inputTensor := ts.MustStack(tensors, 0).MustTo(device, true)
	// inputTensor.Print()

	var (
		output                         *ts.Tensor
		allHiddenStates, allAttentions []ts.Tensor
	)

	ts.NoGrad(func() {
		output, allHiddenStates, allAttentions = model.ForwardT(inputTensor, ts.NewTensor(), ts.NewTensor(), ts.NewTensor(), ts.NewTensor(), false)
	})

	fmt.Printf("output size: %v\n", output.MustSize())

	fmt.Printf("NumHiddenLayers: %v\n", config.NumHiddenLayers)
	fmt.Printf("allHiddenStates length: %v\n", len(allHiddenStates))
	fmt.Printf("allAttentions length: %v\n", len(allAttentions))

}
