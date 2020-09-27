package transformer_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/sugarme/transformer"
)

// No custom params
func TestConfigFromPretrained(t *testing.T) {
	bertURL := transformer.BERTPretrainedConfigs["bert-base-uncased"]

	config := transformer.ConfigFromPretrained(bertURL, nil)
	fmt.Printf("config: %+v\n", config.Params)
	wantVocabSize := 30522
	gotVocabSize := int(config.Params["vocabSize"].(float64))

	if !reflect.DeepEqual(wantVocabSize, gotVocabSize) {
		t.Errorf("Want: %v\n", wantVocabSize)
		t.Errorf("Got: %v\n", gotVocabSize)
	}
}

// With custom params
func TestConfigFromPretrained_CustomParams(t *testing.T) {
	bertURL := transformer.BERTPretrainedConfigs["bert-base-uncased"]

	params := map[string]interface{}{
		"vocabSize": 2000,
	}

	config := transformer.ConfigFromPretrained(bertURL, params)
	fmt.Printf("config: %+v\n", config.Params)
	wantVocabSize := 2000
	gotVocabSize := config.Params["vocabSize"]

	if !reflect.DeepEqual(wantVocabSize, gotVocabSize) {
		t.Errorf("Want: %v\n", wantVocabSize)
		t.Errorf("Got: %v\n", gotVocabSize)
	}
}
