package bert_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/transformer/bert"
)

func TestBertTokenizer(t *testing.T) {
	var tk *bert.Tokenizer = bert.NewTokenizer()
	err := tk.Load("bert-base-uncased", nil)
	if err != nil {
		t.Error(err)
	}

	gotVocabSize := tk.GetVocabSize(false)
	wantVocabSize := 30522

	if !reflect.DeepEqual(wantVocabSize, gotVocabSize) {
		t.Errorf("Want %v\n", wantVocabSize)
		t.Errorf("Got %v\n", gotVocabSize)
	}
}
