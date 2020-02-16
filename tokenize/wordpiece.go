package tokenize

import (
	"fmt"
	"strings"
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

// // Tokenize segments text input into SUBWORD tokens using the supplied vocabulary
// // NOTE: This implementation does not EXACTLY match the ref-impl and behaves slightly differently
// // See https://github.com/google-research/bert/issues/763
// func (wp WordpieceTokenizer) Tokenize(txt string) []string {
// // TODO: determine if utf8 conversion is necessary, per python impl
// // txt = convert_to_unicode(txt)
// var toks, resToks []string
// toks = wp.Basic.Tokenize(txt)
//
// for _, t := range toks {
// resToks = append(resToks, wp.wordpieceTokenizer(t)...)
// }
// return resToks
// }

// func (wp WordpieceTokenizer) wordpieceTokenizer(txt string) []string {
// var toks []string
// for _, tok := range splitWhitespace(txt) {
// if len(tok) > wp.maxWordChars {
// toks = append(toks, wp.unknownToken)
// continue
// }
//
// wordTok := tok
//
// for len(tok) > 0 && tok != "##" {
// var sub string
//
// if tok == wordTok {
// // if tok not in vocab, unknown token
// sub = wp.Basic.Vocab.LongestSubstring(tok)
// if sub == "" {
// toks = append(toks, wp.unknownToken)
// break
// }
//
// // Skip original token
// sub = wp.Basic.Vocab.LongestSubstring(tok[:len(tok)-1])
// toks = append(toks, sub)
//
// tok = fmt.Sprintf("##%s", wordTok[len(sub):])
// fmt.Println(tok)
//
// } else {
// sub = wp.Basic.Vocab.LongestSubstring(tok)
//
// if sub == "" {
// toks = append(toks, wp.unknownToken)
// break
// }
//
// toks = append(toks, sub)
// tok = fmt.Sprintf("##%s", tok[len(sub):])
// fmt.Println(tok)
// }
// }
// }
//
// return toks
// }

func (wp WordpieceTokenizer) Tokenize(txt string) []string {

	// Greedy longest-match-first algorithm
	// Ported from:
	// https://github.com/huggingface/transformers/blob/1eec69a90007b8f4a7af10805dab4904ea5dea77/src/transformers/tokenization_bert.py#L436
	v := wp.Basic.Vocab
	var outputTokens []string

	for _, tok := range splitWhitespace(txt) {
		if len(tok) > wp.maxWordChars {
			outputTokens = append(outputTokens, wp.unknownToken)
			continue
		}

		// if !v.HasToken(tok) {
		// outputTokens = append(outputTokens, wp.unknownToken)
		// continue
		// }

		// wordTok := tok

		isBad := false
		start := 0
		var subTokens []string = []string{}

		chars := strings.Split(tok, "")

		// while loop 1
		// for ok := true; ok; ok = (start < len(chars)) {
		for start < len(chars) {
			end := len(chars)
			var curSubstr string

			// while loop 2
			// for ok := true; ok; ok = (start < end) {
			for start < end {
				substr := strings.Join(chars[start:end], "")
				if start > 0 {
					substr = fmt.Sprintf("##%v", substr)
				}

				if v.HasToken(substr) {
					curSubstr = substr
					break
					// if substr == wordTok { // skip the word token if it is in wordpiece vocab, shouldn't we?
					// end -= 1
					// continue
					// } else {
					// curSubstr = substr
					// break
					// }
				}

				end -= 1

			}

			if curSubstr == "" {
				isBad = true
				break
			}

			subTokens = append(subTokens, curSubstr)

			start = end

		}

		if isBad {
			outputTokens = append(outputTokens, wp.unknownToken)
		}

		outputTokens = append(outputTokens, subTokens...)

	}

	return outputTokens
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
