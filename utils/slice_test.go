package utils_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/sermo/utils"
)

func TestSlice_Insert(t *testing.T) {

	want := []string{"A", "B", "C", "D", "E"}

	var a utils.Slice = utils.Slice{"A", "B", "C", "D", "E"}
	var item string = "x"
	err := a.Insert(2, item)
	if err != nil {
		t.Errorf("%v", err)
	}

	got := a

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}

}
