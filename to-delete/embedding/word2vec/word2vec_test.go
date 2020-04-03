// Package word2vec_test provides test cases for word2vec
package word2vec_test

import (
	"testing"

	"github.com/sugarme/sermo/embedding/word2vec"
)

func TestReadWord2VecBinary(t *testing.T) {

	w2v := word2vec.ReadWord2VecBinary()

	if w2v != nil {
		t.Error("ReadWord2VecBinary should not return anything")
	}
}
