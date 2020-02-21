package document

import (
	"github.com/sugarme/sermo/normalizer"
)

type Offset struct {
	Start int32
	End   int32
}

type Document struct {
	Normalized        NormalizedString
	Ids               []int32
	TypeIds           []int32
	Tokens            []string
	Offsets           []Offset
	SpecialTokensMask []int32
	AttentionMask     []int32
	Overflowing       []Document
}
