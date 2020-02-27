package normalizer_test

import (
	// "fmt"
	"reflect"
	// "strings"
	"testing"
	"unicode"

	// "golang.org/x/text/transform"
	// "golang.org/x/text/unicode/norm"

	"github.com/sugarme/sermo/normalizer"
	// "github.com/sugarme/sermo/utils"
)

func TestNormalized_New(t *testing.T) {

	gotN := normalizer.NewNormalizedFrom("Here you are")
	got := gotN.Get()

	want := normalizer.NormalizedString{
		Original:   "Here you are",
		Normalized: "Here you are",
		Alignments: []normalizer.Alignment{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 10}, {10, 11}, {11, 12}},
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}
}

func isMn(r rune) bool {
	return unicode.Is(unicode.Mn, r) // Mn: nonspacing marks
}

/*
 * func TestNormalized_Transform(t *testing.T) {
 *
 *   wantN := normalizer.NewNormalizedFrom("Here you go déclaré Gülçehre.")
 *   gotN := normalizer.NewNormalizedFrom("Here you go déclaré Gülçehre.")
 *
 *   want := wantN.Get()
 *   gotN.RemoveAccents()
 *   got := gotN.Get()
 *
 *   if !reflect.DeepEqual(want, got) {
 *     t.Errorf("Want: %v\n", want)
 *     t.Errorf("Got: %v\n", got)
 *   }
 *
 * }
 *  */

/*
 * func TestNormalized_NFD(t *testing.T) {
 *
 *   wantN := normalizer.NewNormalizedFrom("déclaré")
 *   gotN := normalizer.NewNormalizedFrom("déclaré")
 *
 *   want := wantN.Get()
 *   gotN.NFD()
 *   got := gotN.Get()
 *
 *   wantASCII := utils.ToASCII(want.Normalized)
 *   gotASCII := utils.ToASCII(got.Normalized)
 *
 *   wantGr := utils.ToGrapheme(want.Normalized)
 *   gotGr := utils.ToGrapheme(want.Normalized)
 *
 *   if !reflect.DeepEqual(want, got) {
 *     t.Errorf("Want: string Form: %s, ASCII string: %v, Grapheme cluster: %v, Bytes: %v (length: %v), Runes: %v\n", want.Normalized, wantASCII, wantGr, []byte(want.Normalized), len(want.Normalized), []rune(want.Normalized))
 *     t.Errorf("Got: string Form: %s, ASCII string: %v, Grapheme cluster: %v, Bytes: %v (length: %v), Runes: %v\n", got.Normalized, gotASCII, gotGr, []byte(got.Normalized), len(got.Normalized), []rune(got.Normalized))
 *   }
 *
 * }
 *  */

func TestNormalized_NFD(t *testing.T) {

	want := []normalizer.Alignment{
		{0, 1},
		{0, 1},
		{1, 2},
		{2, 3},
		{2, 3},
		{3, 4},
		{4, 5},
		{5, 6},
		{6, 7},
	}

	gotN := normalizer.NewNormalizedFrom("élégant")
	beforeAlignments := gotN.Get().Alignments
	beforeNormalized := gotN.Get().Normalized
	gotN.NFD()
	afterAlignments := gotN.Get().Alignments
	afterNormalized := gotN.Get().Normalized

	got := gotN.Get().Alignments

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Alignments Before NFD: %v\n", beforeAlignments)
		t.Errorf("Alignments After NFD: %v\n", afterAlignments)

		t.Errorf("Normalized Before NFD: %v\n", beforeNormalized)
		t.Errorf("Normalized After NFD: %v\n", afterNormalized)

		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}

}

func TestNormalized_NFC(t *testing.T) {

	want := []normalizer.Alignment{
		{0, 1},
		{0, 1},
		{1, 2},
		{2, 3},
		{2, 3},
		{3, 4},
		{4, 5},
		{5, 6},
		{6, 7},
	}

	gotN := normalizer.NewNormalizedFrom("e\u0301le\u0301gant")
	// gotN := normalizer.NewNormalizedFrom("élégantÅ")
	beforeAlignments := gotN.Get().Alignments
	beforeNormalized := gotN.Get().Normalized
	gotN.NFC()
	afterAlignments := gotN.Get().Alignments
	afterNormalized := gotN.Get().Normalized

	got := gotN.Get().Alignments

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Alignments Before NFC: %v\n", beforeAlignments)
		t.Errorf("Alignments After NFC: %v\n", afterAlignments)

		t.Errorf("Normalized Before NFC: %v\n", beforeNormalized)
		t.Errorf("Normalized After NFC: %v\n", afterNormalized)

		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}

}

func TestNormalized_Filter(t *testing.T) {
	gotN := normalizer.NewNormalizedFrom("élégant")

	// gotN.Filter(func(r rune) bool {
	// return r == '\u0301' // single quote for rune literal
	// })

	gotN.Filter('é')
	want := []normalizer.Alignment{
		{0, 1},
		{1, 2},
		{2, 3},
		{3, 4},
		{4, 5},
		{5, 6},
		{6, 7},
	}
	got := gotN.Get().Alignments

	// got := gotN.Get().Normalized
	// want := "élégant"

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}

}
