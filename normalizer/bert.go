package normalizer

import (
	"fmt"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

type BertNormalizer struct {
	DefaultNormalizer       // is BertNormalizer inherits DefaultNormalizer?
	CleanText          bool // Whether to remove Control characters and all sorts of whitespaces replaced with single ` ` space
	Lowercase          bool // Whether to do lowercase
	HandleChineseChars bool // Whether to put spaces around chinese characters so they get split
	StripAccents       bool // whether to remove accents
}

func NewBertNormalizer() BertNormalizer {
	return BertNormalizer{
		CleanText:          true,
		Lowercase:          true,
		HandleChineseChars: false,
		StripAccents:       false,
	}
}

func (bn BertNormalizer) Normalize(txt string) string {
	fmt.Println("Implementing BERT normalizer...")
	// First, Default normalization
	txt = bn.DefaultNormalizer.Normalize(txt)

	// Then, BERT specific normalization
	if bn.CleanText {
		txt = cleanText(txt)
	}

	if bn.Lowercase {
		txt = strings.ToLower(txt)
	}

	if bn.HandleChineseChars {
		txt = padChinese(txt)
	}

	return txt
}

func removeWhitespace(txt string) string {
	return txt
}

func handleControl(txt string) string {
	return txt
}

func handleChineseChar(txt string) string {
	return txt
}

// _Bp is the BERT extension of the Punctuation character range
var _Bp = &unicode.RangeTable{
	R16: []unicode.Range16{
		{0x0021, 0x002f, 1}, // 33-47
		{0x003a, 0x0040, 1}, // 58-64
		{0x005b, 0x0060, 1}, // 91-96
		{0x007b, 0x007e, 1}, // 123-126
	},
	LatinOffset: 4, // All less than 0x00FF
}

var _Bcjk = &unicode.RangeTable{
	R16: []unicode.Range16{
		{0x4e00, 0x9fff, 1},
		{0x3400, 0x4dbf, 1},
		{0xf900, 0xfaff, 1},
	},
	R32: []unicode.Range32{ //govet reports errors on unkeyed fields, but only for this range...
		{Lo: 0x20000, Hi: 0x2a6df, Stride: 1},
		{Lo: 0x2a700, Hi: 0x2b73f, Stride: 1},
		{Lo: 0x2b740, Hi: 0x2b81f, Stride: 1},
		{Lo: 0x2b820, Hi: 0x2ceaf, Stride: 1},
		{Lo: 0x2f800, Hi: 0x2fa1f, Stride: 1},
	},
}

// IsWhitespace checks whether rune c is a BERT whitespace character
func isWhitespace(c rune) bool {
	switch c {
	case ' ':
		return true
	case '\t':
		return true
	case '\n':
		return true
	case '\r':
		return true
	}
	return unicode.Is(unicode.Zs, c)
}

// IsControl checks wher rune c is a BERT control character
func isControl(c rune) bool {
	switch c {
	case '\t':
		return false
	case '\n':
		return false
	case '\r':
		return false
	}
	return unicode.In(c, unicode.Cc, unicode.Cf)
}

// IsPunctuation checks whether rune c is a BERT punctuation character
func isPunctuation(c rune) bool {
	return unicode.In(c, _Bp, unicode.P)
}

// IsChinese validates that rune c is in the CJK range according to BERT spec
func isChinese(c rune) bool {
	return unicode.In(c, _Bcjk, unicode.P)
}

func cleanText(text string) string {
	var b strings.Builder
	for _, c := range text {
		if c == 0 || c == 0xfffd || isControl(c) {
			continue
		} else if isWhitespace(c) {
			b.WriteRune(' ')
		} else {
			b.WriteRune(c)
		}
	}
	return b.String()
}

func stripAccents(text string) string {
	// TODO test
	var b strings.Builder
	for _, c := range norm.NFD.String(text) {
		if !unicode.Is(unicode.Mn, c) {
			b.WriteRune(c)
		}
	}
	return b.String()
}

//padChinese will add space padding around all CJK chars
// This implementation matches BasicTokenizer._tokenize_chinese_chars
func padChinese(text string) string {
	var b strings.Builder
	for _, c := range text {
		if isChinese(c) {
			b.WriteRune(' ')
			b.WriteRune(c)
			b.WriteRune(' ')
		} else {
			b.WriteRune(c)
		}
	}
	return b.String()
}
