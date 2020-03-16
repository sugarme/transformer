// Test cases for package vocab
package vocab_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/sugarme/sermo/vocab"
)

func TestNew(t *testing.T) {
	tokens := map[string]vocab.Idx{
		"apple":  0,
		"banana": 1,
	}

	var want = struct {
		tokens map[string]vocab.Idx
	}{tokens: tokens}

	dict := vocab.New([]string{"apple", "banana"})

	v := reflect.ValueOf(dict)
	got := v.FieldByName("tokens")

	if reflect.DeepEqual(got, want) {
		t.Error("New Dict should create a dict")
	}
}

func TestFromFile(t *testing.T) {
	tokens := map[string]vocab.Idx{
		"apple":  0,
		"banana": 1,
	}

	var want = struct {
		tokens map[string]vocab.Idx
	}{tokens: tokens}

	// TODO(TT): better separate unit test to test scan data to dict
	// from testing reading file system which `integration` test.
	// Ref: https://stackoverflow.com/questions/56397240
	dict, err := vocab.FromFile("mock-data.txt")
	if err != nil {
		fmt.Println(err)
	}

	v := reflect.ValueOf(dict)
	got := v.FieldByName("tokens")

	if err != nil || reflect.DeepEqual(got, want) {
		t.Errorf("Unexpected value: \n Want: %v\n Got: %v\n", want, got)
	}
}

func TestVocab_Add(t *testing.T) {
	want := map[string]vocab.Idx{
		"apple": 0,
	}

	// TODO(TT): better way to initialize dict not using `New` method
	// to separate concern - only test method `Add`.
	var got = vocab.New([]string{})

	err := got.Add("apple")
	if err != nil || reflect.DeepEqual(want, got) {
		t.Errorf("Unexpected Value.\n Want: %v\n, Got: %v\n", want, got)
	}

}

func TestVocab_Index(t *testing.T) {
	want := vocab.Idx(0)

	var vocab = vocab.New([]string{"apple"})

	got, err := vocab.Index("apple")
	if err != nil {
		fmt.Println(err)
	}

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Unexpected Value.\n Want: %v\n, Got: %v\n", want, got)
	}

}

func TestVocab_Token(t *testing.T) {
	var want string = fmt.Sprintf("apple")

	var vocab = vocab.New([]string{"apple"})

	got := vocab.Token(0)

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Unexpected Value.\n Want: %v\n Got: %v\n", want, got)
	}

}

func TestVocab_HasIdx(t *testing.T) {

	var want bool = true

	var vocab = vocab.New([]string{"apple"})

	got := vocab.HasIdx(0)

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Unexpected Value.\n Want: %v\n Got: %v\n", want, got)
	}

}

func TestVocab_HasToken(t *testing.T) {

	var want bool = true

	var vocab = vocab.New([]string{"apple"})

	got := vocab.HasToken("apple")

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Unexpected Value.\n Want: %v\n Got: %v\n", want, got)
	}

}

func TestVocab_Int32(t *testing.T) {
	var want vocab.Idx = 32

	got := vocab.Idx(32)

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Unexpected Value.\n Want: %v\n Got: %v\n", want, got)
	}

}
