// Package bert_test provides test cases for bert package
package bert_test

import (
	"testing"

	"github.com/sugarme/sermo/model/bert"
)

func TestNewBert(t *testing.T) {

	b := bert.NewBert()

	if b != nil {
		t.Error("NewBert func should not return anything")
	}
}
