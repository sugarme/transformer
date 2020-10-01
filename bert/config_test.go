package bert_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/transformer/bert"
)

// No custom params
func TestNewBertConfig_Default(t *testing.T) {

	config := bert.NewConfig(nil)

	wantHiddenAct := "gelu"
	gotHiddenAct := config.HiddenAct
	if !reflect.DeepEqual(wantHiddenAct, gotHiddenAct) {
		t.Errorf("Want: '%v'\n", wantHiddenAct)
		t.Errorf("Got: '%v'\n", gotHiddenAct)
	}

	wantVocabSize := int64(30522)
	gotVocabSize := config.VocabSize

	if !reflect.DeepEqual(wantVocabSize, gotVocabSize) {
		t.Errorf("Want: '%v'\n", wantVocabSize)
		t.Errorf("Got: '%v'\n", gotVocabSize)
	}
}

// With custom params
func TestNewBertConfig_Custom(t *testing.T) {

	config := bert.NewConfig(map[string]interface{}{"VocabSize": int64(2000), "HiddenAct": "relu"})

	wantHiddenAct := "relu"
	gotHiddenAct := config.HiddenAct
	if !reflect.DeepEqual(wantHiddenAct, gotHiddenAct) {
		t.Errorf("Want: '%v'\n", wantHiddenAct)
		t.Errorf("Got: '%v'\n", gotHiddenAct)
	}

	wantVocabSize := int64(2000)
	gotVocabSize := config.VocabSize

	if !reflect.DeepEqual(wantVocabSize, gotVocabSize) {
		t.Errorf("Want: '%v'\n", wantVocabSize)
		t.Errorf("Got: '%v'\n", gotVocabSize)
	}
}
