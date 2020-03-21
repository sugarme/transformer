package pretokenizer_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/sermo/pretokenizer"
)

func TestBytesChar(t *testing.T) {

	want := "!"

	bc := pretokenizer.BytesChar()
	got := bc[33]

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}

}
