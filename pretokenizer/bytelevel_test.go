package pretokenizer_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/sermo/pretokenizer"
)

func TestBytesChar(t *testing.T) {

	want := "!"

	bc := pretokenizer.GenerateBytesChar()
	got := bc[33]

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}

	// Test generating bytesChar map
	bcGot := pretokenizer.BytesChar[33]
	// bcWant := make(map[uint8]string)
	bcWant := "!"
	if !reflect.DeepEqual(bcWant, bcGot) {
		t.Errorf("Want: %v\n", bcWant)
		t.Errorf("Got: %v\n", bcGot)
	}

	// Testing generate charBytes map
	cbGot := pretokenizer.CharBytes["!"]
	var cbWant uint8 = 33
	if !reflect.DeepEqual(cbWant, cbGot) {
		t.Errorf("Want: %v\n", cbWant)
		t.Errorf("Got: %v\n", cbGot)
	}

}
