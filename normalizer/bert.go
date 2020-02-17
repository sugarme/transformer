package normalizer

import (
	"fmt"
)

type BertNormalizer struct {
}

func NewBertNormalizer() {
	fmt.Println("New Bert Normalizer...")
}

func (bn BertNormalizer) Normalize(txt string) string {
	fmt.Println("Implementing BERT normalizer...")
	return txt
}
