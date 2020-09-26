package bert_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/transformer/bert"
)

// No custom params
func TestNewBertConfig_Default(t *testing.T) {

	config := bert.NewBertConfig(nil)

	gotModelType := config.ModelType
	wantModelType := "bert"
	if !reflect.DeepEqual(wantModelType, gotModelType) {
		t.Errorf("Want: '%v'\n", wantModelType)
		t.Errorf("Got: '%v'\n", gotModelType)
	}

	wantHiddenAct := "gelu"
	gotHiddenAct, ok := config.Params["hiddenAct"]
	if !ok {
		t.Errorf("Missing hiddenAct param.\n")
	}
	if !reflect.DeepEqual(wantHiddenAct, gotHiddenAct) {
		t.Errorf("Want: '%v'\n", wantHiddenAct)
		t.Errorf("Got: '%v'\n", gotHiddenAct)
	}

	wantVocabSize := 30522
	gotVocabSize, ok := config.Params["vocabSize"]
	if !ok {
		t.Errorf("Missing vocabSize param.\n")
	}

	if !reflect.DeepEqual(wantVocabSize, gotVocabSize) {
		t.Errorf("Want: '%v'\n", wantVocabSize)
		t.Errorf("Got: '%v'\n", gotVocabSize)
	}
}

// With custom params
func TestNewBertConfig_Custom(t *testing.T) {

	config := bert.NewBertConfig(map[string]interface{}{"vocabSize": 2000, "hiddenAct": "relu"})

	wantHiddenAct := "relu"
	gotHiddenAct, ok := config.Params["hiddenAct"]
	if !ok {
		t.Errorf("Missing hiddenAct param.\n")
	}
	if !reflect.DeepEqual(wantHiddenAct, gotHiddenAct) {
		t.Errorf("Want: '%v'\n", wantHiddenAct)
		t.Errorf("Got: '%v'\n", gotHiddenAct)
	}

	wantVocabSize := 2000
	gotVocabSize, ok := config.Params["vocabSize"]
	if !ok {
		t.Errorf("Missing vocabSize param.\n")
	}

	if !reflect.DeepEqual(wantVocabSize, gotVocabSize) {
		t.Errorf("Want: '%v'\n", wantVocabSize)
		t.Errorf("Got: '%v'\n", gotVocabSize)
	}
}
