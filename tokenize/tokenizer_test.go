package tokenize_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/sugarme/sermo/tokenize"
	"github.com/sugarme/sermo/vocab"
)

func TestBasic(t *testing.T) {
	for _, test := range []struct {
		name   string
		lower  bool
		text   string
		tokens []string
	}{
		{"chinese", false, "ah\u535A\u63A8zz", []string{"ah", "\u535A", "\u63A8", "zz"}},
		{"lower multi", true, " \tHeLLo!how  \n Are yoU?  ", []string{"hello", "!", "how", "are", "you", "?"}},
		{"lower single", true, "H\u00E9llo", []string{"hello"}},
		{"no lower multi", false, " \tHeLLo!how  \n Are yoU?  ", []string{"HeLLo", "!", "how", "Are", "yoU", "?"}},
		{"no lower single", false, "H\u00E9llo", []string{"H\u00E9llo"}},
	} {
		tkz := tokenize.NewTokenizer(test.lower)
		toks := tkz.Tokenize(test.text)
		if !reflect.DeepEqual(toks, test.tokens) {
			t.Errorf("Test %s - Invalid Tokenization - Want: %v, Got: %v", test.name, test.tokens, toks)
		}
	}
}

func TestWordpiece(t *testing.T) {
	voc := vocab.New([]string{"[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing", "do"})
	// voc, err := vocab.FromFile("vocab.txt")
	// if err != nil {
	// t.Fatal(err)
	// }

	fmt.Printf("Vocab Size: %v\n", voc.Size())
	// idx, _ := voc.Index("##un")
	// t.Fatal(fmt.Sprintf("Index of '##un': %v\n", idx))

	var options = []tokenize.Option{
		tokenize.WithWordpieceTokenizer(tokenize.DefaultMaxWordChars, tokenize.DefaultUnknownToken, voc),
		// tokenize.WithVocab(voc),
	}

	for i, test := range []struct {
		text   string
		tokens []string
	}{
		{"", nil},
		{"unwanted", []string{"un", "##want", "##ed"}},
		{"unwanted running", []string{"un", "##want", "##ed", "runn", "##ing"}},
		// {"unwantedX", []string{"[UNK]"}},
		//{"unwantedX running", []string{"[UNK]", "runn", "##ing"}},
	} {
		tkz := tokenize.NewTokenizer(true, options...)

		toks := tkz.Tokenize(test.text)

		if !reflect.DeepEqual(toks, test.tokens) {
			t.Errorf("Test %d - Input: %s - Invalid Tokenization - Want: %v, Got: %v", i, test.text, test.tokens, toks)
		}
	}
}
