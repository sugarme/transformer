package roberta_test

import (
	// "fmt"
	"log"
	"reflect"
	"testing"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/model/bpe"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	"github.com/sugarme/tokenizer/processor"
	"github.com/sugarme/transformer/bert"
	"github.com/sugarme/transformer/roberta"
)

func getRobertaTokenizer() (retVal *tokenizer.Tokenizer) {
	vocabFile := "../data/roberta/roberta-base-vocab.json"
	mergeFile := "../data/roberta/roberta-base-merges.txt"

	model, err := bpe.NewBpeFromFiles(vocabFile, mergeFile)
	if err != nil {
		log.Fatal(err)
	}

	tk := tokenizer.NewTokenizer(model)
	// fmt.Printf("Vocab size: %v\n", tk.GetVocabSize(false))

	bertNormalizer := normalizer.NewBertNormalizer(true, true, true, true)
	tk.WithNormalizer(bertNormalizer)

	blPreTokenizer := pretokenizer.NewByteLevel()
	// blPreTokenizer.SetAddPrefixSpace(false)
	tk.WithPreTokenizer(blPreTokenizer)

	var specialTokens []tokenizer.AddedToken
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<s>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<pad>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("</s>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<unk>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<mask>", true))
	tk.AddSpecialTokens(specialTokens)

	postProcess := processor.DefaultRobertaProcessing()
	tk.WithPostProcessor(postProcess)

	return tk
}

func getRobertaModel() {

}

func TestRobertaForMaskedLM(t *testing.T) {

	// Config
	config := new(bert.BertConfig)
	err := config.Load("../data/roberta/roberta-base-config.json", nil)
	if err != nil {
		log.Fatal(err)
	}

	// Model
	device := gotch.CPU
	vs := nn.NewVarStore(device)

	model := roberta.NewRobertaForMaskedLM(vs.Root(), config)
	err = vs.Load("../data/roberta/roberta-base-model.gt")
	if err != nil {
		log.Fatal(err)
	}

	// Roberta tokenizer
	tk := getRobertaTokenizer()

	sentence1 := "Looks like one <mask> is missing"
	sentence2 := "It's like comparing <mask> to apples"

	input := []tokenizer.EncodeInput{
		tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(sentence1)),
		tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(sentence2)),
	}

	/*
	 *   // NOTE: EncodeBatch does encode concurrently, so it does not keep original order!
	 *   encodings, err := tk.EncodeBatch(input, true)
	 *   if err != nil {
	 *     log.Fatal(err)
	 *   }
	 *  */
	var encodings []tokenizer.Encoding
	for _, i := range input {
		en, err := tk.Encode(i, true)
		if err != nil {
			log.Fatal(err)
		}
		encodings = append(encodings, *en)
	}

	// fmt.Printf("encodings:\n%+v\n", encodings)

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

		tensors = append(tensors, ts.TensorFrom(tokInput))
	}

	inputTensor := ts.MustStack(tensors, 0).MustTo(device, true)

	var output ts.Tensor
	ts.NoGrad(func() {
		output, _, _, err = model.Forward(inputTensor, ts.None, ts.None, ts.None, ts.None, ts.None, ts.None, false)
		if err != nil {
			log.Fatal(err)
		}
	})

	index1 := output.MustGet(0).MustGet(4).MustArgmax(0, false, false).Int64Values()[0]
	index2 := output.MustGet(1).MustGet(5).MustArgmax(0, false, false).Int64Values()[0]
	gotMask1 := tk.Decode([]int{int(index1)}, false)
	gotMask2 := tk.Decode([]int{int(index2)}, false)

	// fmt.Printf("index1: '%v' - mask1: '%v'\n", index1, gotMask1)
	// fmt.Printf("index2: '%v' - mask2: '%v'\n", index2, gotMask2)
	wantMask1 := "Ġperson"
	wantMask2 := "Ġapples"

	if !reflect.DeepEqual(wantMask1, gotMask1) {
		t.Errorf("Want: %v got %v\n", wantMask1, gotMask1)
	}

	if !reflect.DeepEqual(wantMask2, gotMask2) {
		t.Errorf("Want: %v got %v\n", wantMask2, gotMask2)
	}

}
