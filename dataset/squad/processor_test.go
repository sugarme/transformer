package squad_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/sugarme/tokenizer/pretrained"
	"github.com/sugarme/transformer/dataset/squad"
)

func TestConvertExampleToFeatures(t *testing.T) {

	tk := pretrained.BertLargeCasedWholeWordMaskingSquad()

	examples := squad.LoadV2("dev") // dataset shape: [4 20302 384]
	// examples := squad.LoadV2("train") // dataset shape: [4 86821 384]
	features, dataset := squad.ConvertExamplesToFeatures(examples, tk, "bert", 384, 128, 64, "[SEP]", "[PAD]", 0, false, true)

	fmt.Printf("num of features: %v\n", len(features))
	fmt.Printf("dataset shape: %v\n", dataset.MustSize())

	wantMaxSeqLen := int64(384)
	gotMaxSeqLen := dataset.MustSize()[2]

	if !reflect.DeepEqual(wantMaxSeqLen, gotMaxSeqLen) {
		t.Errorf("Want max sequence len: %v\n", wantMaxSeqLen)
		t.Errorf("Got max sequence len: %v\n", gotMaxSeqLen)
	}
}
