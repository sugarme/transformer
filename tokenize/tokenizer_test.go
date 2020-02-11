// Test cases for package tonenize
package tokenize_test

import (
	"testing"

	"github.com/sugarme/sermo/tokenize"
)

func TestNew(t *testing.T) {

	tk := tokenize.New()

	if tk == nil {
		t.Error("func New should not return anything")
	}

}
