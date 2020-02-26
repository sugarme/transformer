package normalizer

import (
	"strings"
	"unicode"

	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

// Basic Unicode normal form composing and decomposing - NFC, NFD, NFKC, NFKD
// Ref. https://blog.golang.org/normalization

// UnicodeNormalizer is a struct that provides basic unicode
// normal form methods
type UnicodeNormalizer struct{}

func (un UnicodeNormalizer) NFC(s string) string {

	t := transform.Chain(norm.NFC)

	normalized, _, _ := transform.String(t, s)

	return normalized

}

func (un UnicodeNormalizer) NFD(s string) string {

	t := transform.Chain(norm.NFD)

	normalized, _, _ := transform.String(t, s)

	return normalized

}

func (un UnicodeNormalizer) NFKC(s string) string {

	t := transform.Chain(norm.NFKC)

	normalized, _, _ := transform.String(t, s)

	return normalized

}

func (un UnicodeNormalizer) NFKD(s string) string {

	t := transform.Chain(norm.NFKD)

	normalized, _, _ := transform.String(t, s)

	return normalized

}

// RemoveAccents decomposes text into small parts, removes all accents
// then recomposes the text into NFC
func (un UnicodeNormalizer) RemoveAccents(s string) string {

	t := transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)

	normalized, _, _ := transform.String(t, s)

	return normalized

}

func isMn(r rune) bool {
	return unicode.Is(unicode.Mn, r) // Mn: nonspacing marks
}

func (un UnicodeNormalizer) ToLower(s string) string {

	return strings.ToLower(s)

}

func isFiltered(r, filtered rune) bool {
	return r == filtered
}
