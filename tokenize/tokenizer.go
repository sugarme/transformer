// Package tokenizer provides various tools for tokenization using different models (BPE, word, unigram)
// It also provides options to include
// 1. normalizer
// 2. pre-processing
// 3. post-processing.
package tokenize

import (
	// "fmt"

	"github.com/sugarme/sermo/vocab"
)

// Tokenizer defines an interface to implement various types (levels) of tokenizer
type Tokenizer interface {
	Tokenize(text string) (tokens []string)
}

type tokenizer struct {
	// Vocab     vocab.Dict
	Tokenizer Tokenizer
}

func (t tokenizer) Tokenize(text string) (tokens []string) {
	return t.Tokenizer.Tokenize(text)
}

var defaultTokenizer = tokenizer{
	// Vocab:     vocab.New([]string{}),
	Tokenizer: NewWordTokenizer(true, vocab.New([]string{})),
}

func newTokenizer(lower bool) tokenizer {
	return tokenizer{
		// Vocab:     vocab.New([]string{}),
		Tokenizer: NewWordTokenizer(lower, vocab.New([]string{})),
	}
}

type Option func(*tokenizer)

// func WithVocab(v vocab.Dict) Option {
// return func(o *tokenizer) {
// o.Vocab = v
// }
// }

func WithWordpieceTokenizer(maxWordChars int, unknownToken string, v vocab.Dict) Option {
	return func(o *tokenizer) {
		o.Tokenizer = WordpieceTokenizer{
			Basic:        NewWordTokenizer(true, v),
			maxWordChars: maxWordChars,
			unknownToken: unknownToken,
		}
	}
}

func NewTokenizer(lower bool, opts ...Option) Tokenizer {
	tkz := newTokenizer(lower)

	for _, o := range opts {
		o(&tkz)
	}

	return tkz
}
