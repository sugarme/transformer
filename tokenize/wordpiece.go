package tokenize

import (
	"fmt"
)

// WordpieceTokenizer provides methods to break down and work on subwords
// Ref. https://arxiv.org/pdf/1609.08144.pdf
type WordpieceTokenizer struct {
	Basic        WordTokenizer
	maxWordChars int
	unknownToken string
}

// DefaultMaxWordChars is the max length of a token for it to be tokenized,
// otherwise marked as unknown
const DefaultMaxWordChars = 200

// DefaultUnknownToken is the token used to signify an unkown token
const DefaultUnknownToken = "[UNK]"

// NewWordpiece returns a WordpieceTokenizer with the default settings.
// Generally should be used in a FullTokenizer
// TODO: add options
func NewWordpieceTokenizer() WordpieceTokenizer {
	return WordpieceTokenizer{
		maxWordChars: DefaultMaxWordChars,
		unknownToken: DefaultUnknownToken,
	}
}

// Tokenize segments text input into SUBWORD tokens using the supplied vocabulary
// NOTE: This implementation does not EXACTLY match the ref-impl and behaves slightly differently
// See https://github.com/google-research/bert/issues/763
func (wp WordpieceTokenizer) Tokenize(txt string) []string {
	// TODO: determine if utf8 conversion is necessary, per python impl
	// txt = convert_to_unicode(txt)
	var toks, resToks []string
	toks = wp.Basic.Tokenize(txt)

	for _, t := range toks {
		resToks = append(resToks, wp.wordpieceTokenizer(t)...)
	}
	return resToks
}

func (wp WordpieceTokenizer) wordpieceTokenizer(txt string) []string {
	var toks []string
	for _, tok := range splitWhitespace(txt) {
		if len(tok) > wp.maxWordChars {
			toks = append(toks, wp.unknownToken)
			continue
		}
		for len(tok) > 0 && tok != "##" {
			sub := wp.Basic.Vocab.LongestSubstring(tok)
			if sub == "" {
				toks = append(toks, wp.unknownToken)
				break
			}
			toks = append(toks, sub)
			tok = fmt.Sprintf("##%s", tok[len(sub):])
		}
	}
	return toks
}

// SetMaxWordChars will set the max chars for a word to be tokenized,
// generally this should be congfigured through the FullTokenizer
func (wp WordpieceTokenizer) SetMaxWordChars(c int) {
	wp.maxWordChars = c
}

// SetUnknownToken will set the , generally this should be congfigured through the FullTokenizer
func (wp WordpieceTokenizer) SetUnknownToken(tok string) {
	wp.unknownToken = tok
}
