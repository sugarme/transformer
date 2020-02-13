package tokenize_test

import (
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
		tkz := tokenize.NewWordTokenizer()
		toks := tkz.Tokenize(test.text)
		if !reflect.DeepEqual(toks, test.tokens) {
			t.Errorf("Test %s - Invalid Tokenization - Want: %v, Got: %v", test.name, test.tokens, toks)
		}
	}
}

func TestWordpiece(t *testing.T) {
	voc := vocab.New([]string{"[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing"})
	for i, test := range []struct {
		text   string
		tokens []string
	}{
		{"", nil},
		{"unwanted", []string{"un", "##want", "##ed"}},
		{"unwanted running", []string{"un", "##want", "##ed", "runn", "##ing"}},
		// TODO determine if these tests are correct
		//	{"unwantedX", []string{"[UNK]"}},
		//{"unwantedX running", []string{"[UNK]", "runn", "##ing"}},
	} {
		tkz := tokenize.NewTokenizer(tokenize.WithWordpieceTokenizer(tokenize.DefaultMaxWordChars, tokenize.DefaultUnknownToken),
			tokenize.WithVocab(voc),
		)
		toks := tkz.Tokenize(test.text)
		if !reflect.DeepEqual(toks, test.tokens) {
			t.Errorf("Test %d - Invalid Tokenization - Want: %v, Got: %v", i, test.tokens, toks)
		}
	}
}
