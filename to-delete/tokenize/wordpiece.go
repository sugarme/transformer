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

// Tokenize dices text input into SUBWORD tokens using the supplied vocabulary
func (wp WordpieceTokenizer) Tokenize(txt string) []string {
	var toks, subToks []string
	toks = wp.Basic.Tokenize(txt)

	for _, tok := range toks {
		if len(tok) > wp.maxWordChars {
			subToks = append(subToks, wp.unknownToken)
			continue
		}

		for len(tok) > 0 && tok != "##" {
			subTok := wp.LongestSubstring(tok)

			if subTok == "" {
				subToks = append(subToks, wp.unknownToken)
				break
			}

			subToks = append(subToks, subTok)

			tok = fmt.Sprintf("##%s", tok[len(subTok):])
		}

	}

	return subToks
}

// SetMaxWordChars sets the max chars for a word to be tokenized,
func (wp WordpieceTokenizer) SetMaxWordChars(c int) {
	wp.maxWordChars = c
}

// SetUnknownToken sets special string to be unknown token
func (wp WordpieceTokenizer) SetUnknownToken(tok string) {
	wp.unknownToken = tok
}

// longestSubstring returns the longest substring of an input token
// by looking up internal vocabular dictionary
// Algorithm: `longest prefix match`
// Ref. https://en.wikipedia.org/wiki/Longest_prefix_match
func (wp WordpieceTokenizer) LongestSubstring(token string) string {
	v := wp.Basic.Vocab
	// Greedy.
	// TODO(if needed): trie (ref. https://en.wikipedia.org/wiki/Trie)
	for i := len(token); i > 0; i-- {
		sub := token[:i]
		if v.HasToken(sub) {
			return sub
		}
	}

	return ""
}
