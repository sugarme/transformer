package normalizer

import (
	"fmt"
)

type BertNormalizer struct {
	DefaultNormalizer // is BertNormalizer inherits DefaultNormalizer?
	IsWhitespace      bool
	IsControl         bool
	IsChineseChar     bool
}

func NewBertNormalizer() BertNormalizer {
	return BertNormalizer{
		IsWhitespace:  true,
		IsControl:     true,
		IsChineseChar: false,
	}
}

func (bn BertNormalizer) Normalize(txt string) string {
	fmt.Println("Implementing BERT normalizer...")
	// First, Default normalization
	txt = bn.DefaultNormalizer.Normalize(txt)

	// Then, BERT specific normalization
	if bn.IsWhitespace {
		txt = removeWhitespace(txt)
	}

	if bn.IsControl {
		txt = handleControl(txt)
	}

	if bn.IsChineseChar {
		txt = handleChineseChar(txt)
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
