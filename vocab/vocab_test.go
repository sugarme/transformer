// Test cases for package vocab
package vocab_test

import (
	"testing"

	"github.com/sugarme/sermo/vocab"
)

func TestNewDict(t *testing.T) {
	voc := vocab.NewDict([]vocab.Token{})

	if len(voc.Tokens) == 0 {
		t.Error("New Dict should return error")
	}
}
