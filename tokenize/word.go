package tokenize

import (
	"fmt"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"

	"github.com/sugarme/sermo/vocab"
)

type WordTokenizer struct {
	Lower bool
	Vocab vocab.Dict
}

func NewWordTokenizer() WordTokenizer {
	return WordTokenizer{
		Lower: true,
	}
}

// Tokenize breaks down input text into word tokens using
// basic process:
// - cleaning,
// - unicode conversion,
// - whitespace spliting,
// - punct spliting
func (w WordTokenizer) Tokenize(txt string) []string {
	txt = clean(txt)
	txt = padChinese(txt)

	fmt.Printf("We are here \n")

	toks := basicTokenize(txt, w.Lower)

	return toks

}

func clean(txt string) string {
	var b strings.Builder
	for _, c := range txt {
		if c == 0 || c == 0xfffd || isControl(c) {
			continue
		} else if isSpace(c) {
			b.WriteRune(' ')
		} else {
			b.WriteRune(c)
		}
	}
	return b.String()
}

func stripAccents(txt string) string {
	var b strings.Builder
	for _, c := range norm.NFD.String(txt) {
		if !unicode.Is(unicode.Mn, c) {
			b.WriteRune(c)
		}
	}
	return b.String()
}

func splitPunct(txt string) []string {
	var toks []string
	var b strings.Builder

	for _, c := range txt {
		if isPunct(c) {
			toks = append(toks, b.String())
			toks = append(toks, string(c))
			b.Reset()
		} else {
			b.WriteRune(c)
		}
	}

	if b.Len() > 0 {
		toks = append(toks, b.String())
	}
	return toks
}

func basicTokenize(txt string, lower bool) []string {

	var toks []string
	toks = splitWhitespace(txt)

	if lower {
		toks = toLower(toks)
	}

	var resToks []string
	for _, tok := range toks {
		subTok := splitPunct(tok)
		resToks = append(resToks, subTok...)
	}

	resToks = splitWhitespace(strings.Join(resToks, " "))
	return resToks
}

func toLower(txt []string) []string {
	var lTxt []string
	for _, t := range txt {
		t = strings.ToLower(t)
		t = stripAccents(t)
		lTxt = append(lTxt, t)
	}

	return lTxt

}

//splitWhitespace splits text into tokens by whitespace
func splitWhitespace(txt string) []string {
	var toks []string = strings.Split(txt, " ")

	// remove empty string
	for _, tok := range toks {
		if tok != "" {
			toks = append(toks, tok)
		}
	}
	return toks

}

//padChinese adds space padding around all CJK chars
func padChinese(txt string) string {
	var b strings.Builder
	for _, c := range txt {
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
