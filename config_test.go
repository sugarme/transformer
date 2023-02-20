package transformer_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/transformer"
	"github.com/sugarme/transformer/bert"
)

// With model name
func TestConfigFromPretrained_ModelName(t *testing.T) {
	modelName := "bert-base-uncased"
	var config *bert.BertConfig = new(bert.BertConfig)
	err := transformer.LoadConfig(config, modelName, nil)
	if err != nil {
		t.Error(err)
	}

	wantVocabSize := int64(30522)
	gotVocabSize := config.VocabSize

	if !reflect.DeepEqual(wantVocabSize, gotVocabSize) {
		t.Errorf("Want: %v\n", wantVocabSize)
		t.Errorf("Got: %v\n", gotVocabSize)
	}
}

// No custom params
func TestConfigFromPretrained(t *testing.T) {
	var config *bert.BertConfig = new(bert.BertConfig)
	err := transformer.LoadConfig(config, "bert-base-uncased", nil)
	if err != nil {
		t.Error(err)
	}

	wantVocabSize := int64(30522)
	gotVocabSize := config.VocabSize

	if !reflect.DeepEqual(wantVocabSize, gotVocabSize) {
		t.Errorf("Want: %v\n", wantVocabSize)
		t.Errorf("Got: %v\n", gotVocabSize)
	}

}

// With custom params
func TestConfigFromPretrained_CustomParams(t *testing.T) {
	params := map[string]interface{}{
		"VocabSize": int64(2000),
		"NumLabels": int64(4),
	}

	var config *bert.BertConfig = new(bert.BertConfig)
	err := transformer.LoadConfig(config, "bert-base-uncased", params)
	if err != nil {
		t.Error(err)
	}

	wantVocabSize := int64(2000)
	gotVocabSize := config.VocabSize

	if !reflect.DeepEqual(wantVocabSize, gotVocabSize) {
		t.Errorf("Want: %v\n", wantVocabSize)
		t.Errorf("Got: %v\n", gotVocabSize)
	}

	wantNumLabels := int64(4)
	gotNumLabels := config.NumLabels

	if !reflect.DeepEqual(wantNumLabels, gotNumLabels) {
		t.Errorf("Want: %v\n", wantNumLabels)
		t.Errorf("Got: %v\n", gotNumLabels)
	}
}
