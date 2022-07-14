package main

import (
	"flag"
	"fmt"
	"log"
	// "strings"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/decoder"
	"github.com/sugarme/tokenizer/model/wordpiece"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	// "github.com/sugarme/tokenizer/pretrained"
	"github.com/sugarme/tokenizer/processor"

	"github.com/sugarme/transformer/bert"
)

var input string

func init() {
	flag.StringVar(&input, "input", "Steve went to Paris", "Enter an input text to make inference.")
}

func main() {
	flag.Parse()

	// Config
	config, err := bert.ConfigFromFile("../../data/bert/config-ner.json")
	if err != nil {
		log.Fatal(err)
	}

	// Model
	device := gotch.CPU
	// device := gotch.NewCuda().CudaIfAvailable()
	vs := nn.NewVarStore(device)

	model := bert.NewBertForTokenClassification(vs.Root(), config)
	err = vs.Load("../../data/bert/bert-ner.gt")
	if err != nil {
		log.Fatalf("Load model weight error: \n%v", err)
	}

	fmt.Printf("Varstore weights have been loaded\n")
	fmt.Printf("Num of variables: %v\n", len(vs.Variables()))

	tk := nerTokenizer()

	// Labels
	labelMap := map[int64]string{
		1:  "O",      // Outside of the named entity
		2:  "B-MISC", // Beginning of a miscellaneous entity right after another miscellaneous entity
		3:  "I-MISC", // Miscellaneous entity
		4:  "B-PER",  // Beginning of a person's name right after another person's name
		5:  "I-PER",  //  Person's name
		6:  "B-ORG",  // Beginning of a organisation right after another org
		7:  "I-ORG",  // Organisation
		8:  "B-LOC",  // Beginning of a location right after another location.
		9:  "I-LOC",  // Location
		10: "[CLS]",  // CLS token
		11: "[SEP]",  // SEP token
	}

	encoding, err := tk.EncodeSingle(input, true)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%+v\n", encoding)

	var batchSize int64 = 1
	var seqLen int64 = int64(len(encoding.Ids))

	// inputIds
	inputTensor := ts.MustOfSlice(toInt64(encoding.Ids)).MustTo(device, true).MustView([]int64{batchSize, seqLen}, true)
	// segmentIds
	tokenTypeIds := ts.MustOfSlice(toInt64(encoding.TypeIds)).MustTo(device, true).MustView([]int64{batchSize, seqLen}, true)
	// inputMasks
	attentionMask := ts.MustOfSlice(toInt64(encoding.AttentionMask)).MustTo(device, true).MustView([]int64{batchSize, seqLen}, true)
	// specialTokenMask
	specialTokenMask := ts.MustOfSlice(toInt64(encoding.SpecialTokenMask)).MustTo(device, true).MustView([]int64{batchSize, seqLen}, true)

	var output *ts.Tensor
	ts.NoGrad(func() {
		output, _, _ = model.ForwardT(inputTensor, tokenTypeIds, specialTokenMask, attentionMask, ts.None, false)
	})

	fmt.Printf("%i\n", output)
	logits := output.MustSoftmax(2, gotch.Double, false)
	labelIds := logits.MustArgmax([]int64{2}, true, false).Int64Values()

	fmt.Printf("Logits:\n%0.5f\n", logits)
	fmt.Printf("Logits Label:\n%v\n", labelIds)

	fmt.Printf("Input: %q\n", input)
	for i, token := range encoding.Tokens {
		if token != "[CLS]" && token != "[SEP]" {
			pvalues := logits.MustSqueeze1(0, false)
			selectIdx := ts.NewSelect(int64(i))
			pVals := pvalues.Idx(selectIdx).Float64Values()
			labelId := labelIds[i]
			fmt.Printf("%-10s (%-5s, p=%.5f)\n", token, labelMap[labelId], max(pVals))
		}
	}
}

func toInt64(data []int) []int64 {
	var data64 []int64
	for _, v := range data {
		data64 = append(data64, int64(v))
	}

	return data64
}

func filterPosition(data []int) []int {
	var filterData []int
	for _, v := range data {
		if v != -1 {
			filterData = append(filterData, v)
		}
	}
	return filterData
}

func nerTokenizer() *tokenizer.Tokenizer {
	vocabFile := "../../data/bert/vocab-ner.txt"
	model, err := wordpiece.NewWordPieceFromFile(vocabFile, "[UNK]")
	if err != nil {
		log.Fatal(err)
	}

	tk := tokenizer.NewTokenizer(model)

	// don't do lowercase
	bertNormalizer := normalizer.NewBertNormalizer(true, false, true, true)
	tk.WithNormalizer(bertNormalizer)

	bertPreTokenizer := pretokenizer.NewBertPreTokenizer()
	tk.WithPreTokenizer(bertPreTokenizer)

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

	tk.AddSpecialTokens([]tokenizer.AddedToken{tokenizer.NewAddedToken("[MASK]", true)})
	tk.AddSpecialTokens([]tokenizer.AddedToken{tokenizer.NewAddedToken("[SEP]", true)})
	tk.AddSpecialTokens([]tokenizer.AddedToken{tokenizer.NewAddedToken("[CLS]", true)})

	wpDecoder := decoder.DefaultWordpieceDecoder()
	tk.WithDecoder(wpDecoder)

	return tk
}

func max(vals []float64) float64 {
	max := 0.0
	for _, v := range vals {
		if v > max {
			max = v
		}
	}

	return max
}
