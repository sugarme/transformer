// Package tokenize provides various tools for tokenization
// including word dicing and text segmentation.
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
	Vocab     vocab.Dict
	Tokenizer Tokenizer
}

func (t tokenizer) Tokenize(text string) (tokens []string) {
	return nil
}

var defaultTokenizer = tokenizer{
	Vocab:     vocab.New([]string{}),
	Tokenizer: WordTokenizer{Lower: true},
}

type Option func(*tokenizer)

func WithVocab(vocab vocab.Dict) Option {
	return func(o *tokenizer) {
		o.Vocab = vocab
	}
}

func WithWordpieceTokenizer(maxWordChars int, unknownToken string) Option {
	return func(o *tokenizer) {
		o.Tokenizer = WordpieceTokenizer{
			Basic:        WordTokenizer{Lower: true},
			maxWordChars: maxWordChars,
			unknownToken: unknownToken,
		}
	}
}

func NewTokenizer(opts ...Option) Tokenizer {
	tkz := defaultTokenizer

	for _, o := range opts {
		o(&tkz)
	}

	return tkz
}
