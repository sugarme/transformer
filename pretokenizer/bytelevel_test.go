package pretokenizer_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/sugarme/sermo/normalizer"
	"github.com/sugarme/sermo/pretokenizer"
	"github.com/sugarme/sermo/tokenizer"
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

func TestDecoding(t *testing.T) {

	bytelevel := pretokenizer.NewByteLevel()
	bytelevel.SetAddPrefixSpace(false)

	want := "Hello my friend, how is your day going?"

	toks := []string{
		"Hello",
		"Ġmy",
		"Ġfriend",
		",",
		"Ġhow",
		"Ġis",
		"Ġyour",
		"Ġday",
		"Ġgoing",
		"?",
	}

	got := bytelevel.Decode(toks)

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}
}

func TestAddPrefixSpace(t *testing.T) {

	bytelevel := pretokenizer.NewByteLevel()
	bytelevel.SetAddPrefixSpace(true)

	lines := []string{
		" Hello my friend, how is your day going?",
		"Hello my friend, how is your day going?",
	}

	for _, l := range lines {
		var normalized *normalizer.Normalized
		normalized = normalizer.NewNormalizedFrom(l)

		normalized, res := bytelevel.PreTokenize(normalized)

		nwant := "ĠHelloĠmyĠfriend,ĠhowĠisĠyourĠdayĠgoing?"
		ngot := normalized.GetNormalized()

		pwant := []pretokenizer.PreTokResult{
			{Content: "ĠHello", Offsets: tokenizer.Offsets{Start: 0, End: 6}},
			{Content: "Ġmy", Offsets: tokenizer.Offsets{Start: 6, End: 9}},
			{Content: "Ġfriend", Offsets: tokenizer.Offsets{Start: 9, End: 16}},
			{Content: ",", Offsets: tokenizer.Offsets{Start: 16, End: 17}},
			{Content: "Ġhow", Offsets: tokenizer.Offsets{Start: 17, End: 21}},
			{Content: "Ġis", Offsets: tokenizer.Offsets{Start: 21, End: 24}},
			{Content: "Ġyour", Offsets: tokenizer.Offsets{Start: 24, End: 29}},
			{Content: "Ġday", Offsets: tokenizer.Offsets{Start: 29, End: 33}},
			{Content: "Ġgoing", Offsets: tokenizer.Offsets{Start: 33, End: 39}},
			{Content: "?", Offsets: tokenizer.Offsets{Start: 39, End: 40}},
		}

		pgot := *res

		if !reflect.DeepEqual(nwant, ngot) {
			t.Errorf("nWant: %v\n", nwant)
			t.Errorf("nGot: %v\n", ngot)
		}

		if !reflect.DeepEqual(pwant, pgot) {
			t.Errorf("pWant: %v\n", pwant)
			t.Errorf("pGot: %v\n", pgot)
		}
	}

}

func TestDecodeWorksOnSeparatedTokens(t *testing.T) {

	bytelevel := pretokenizer.NewByteLevel()
	bytelevel.SetAddPrefixSpace(false)

	lines := []string{
		// "A Nuskhuri abbreviation of იესუ ქრისტე ( iesu kriste ) \" Jesus Christ \"",
		// "An equal number have descenders , like p or q in English : გ , დ , ე , ვ , კ , ლ , ჟ , ტ , უ , ფ , ღ , ყ , ც",
		"aგ w",
	}

	for _, l := range lines {
		var normalized *normalizer.Normalized
		normalized = normalizer.NewNormalizedFrom(l)

		_, preTokenized := bytelevel.PreTokenize(normalized)

		var separatedTokens []string
		for _, preTok := range *preTokenized {
			chars := strings.Split(preTok.Content, "")
			separatedTokens = append(separatedTokens, chars...)
		}

		want := l
		got := bytelevel.Decode(separatedTokens)

		if !reflect.DeepEqual(want, got) {
			t.Errorf("Want: %v\n", want)
			t.Errorf("Got: %v\n", got)
		}
	}
}
