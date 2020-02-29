package normalizer

import (
	"golang.org/x/text/unicode/norm"
)

// Basic Unicode normal form composing and decomposing - NFC, NFD, NFKC, NFKD
// Ref. https://blog.golang.org/normalization

// type NFD struct{}
//
// func (un *NFD) Normalize(n Normalized) (Normalized, error) {
// n.NFD()
//
// return n, nil
// }
//
// type NFC struct{}
//
// func (un *NFC) Normalize(n Normalized) (Normalized, error) {
// n.NFC()
//
// return n, nil
// }
//
// type NFKD struct{}
//
// func (un *NFKD) Normalize(n Normalized) (Normalized, error) {
// n.NFKD()
//
// return n, nil
// }
//
// type NFKC struct{}
//
// func (un *NFKC) Normalize(n Normalized) (Normalized, error) {
// n.NFKC()
//
// return n, nil
// }

type UnicodeNormalizer struct {
	Form norm.Form
}

func NewUnicodeNormalizer(form norm.Form) UnicodeNormalizer {
	return UnicodeNormalizer{
		Form: form,
	}
}

func (un *UnicodeNormalizer) Normalize(n Normalized) (Normalized, error) {
	switch un.Form {
	case norm.NFC:
		n.NFC()
	case norm.NFD:
		n.NFD()
	case norm.NFKC:
		n.NFKC()
	case norm.NFKD:
		n.NFKD()
	}

	return n, nil
}
