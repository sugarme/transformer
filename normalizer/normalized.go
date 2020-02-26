package normalizer

import (
	"errors"
	"fmt"
	"log"
	"strings"

	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"

	"github.com/sugarme/sermo/utils"
)

// NormalizedString keeps both versions of an input string and
// provides methods to access them
type NormalizedString struct {
	Original   string
	Normalized string
	Alignments []Alignment
}

// Alignment maps normalized string to original one using `rune` (Unicode code point)
// Pos: the rune position in the modified (normalized) string and
// Changes: representing the number (size) of inserted/deleted runes from original string
type Alignment struct {
	Pos     int
	Changes int
}

// Normalized is wrapper for a `NormalizedString` and provides
// methods to access it.
type Normalized struct {
	normalizedString NormalizedString
}

// NewNormalizedFrom creates a Normalized instance from string input
func NewNormalizedFrom(s string) Normalized {
	var alignments []Alignment

	// Break down string to slice of runes
	for i, _ := range []rune(s) {
		alignments = append(alignments, Alignment{
			Pos:     i,
			Changes: i + 1,
		})
	}

	n := NormalizedString{
		Original:   s,
		Normalized: s,
		Alignments: alignments,
	}

	return Normalized{
		normalizedString: n,
	}

}

func (n *Normalized) Get() NormalizedString {
	return n.normalizedString
}

func (n *Normalized) GetNormalized() string {
	return n.normalizedString.Normalized
}

func (n *Normalized) GetOriginal() string {
	return n.normalizedString.Original
}

// OriginalOffsets returns the range of the original string corresponding to
// the received range on the normalized string.
// Returns None if out of bounds
func (n *Normalized) OriginalOffsets(r []int) []int {
	start := r[0]
	end := len(r)

	var selectedAlignments []Alignment

	firstAlign := n.normalizedString.Alignments[0]
	lastAlign := n.normalizedString.Alignments[len(n.normalizedString.Alignments)]

	if start < firstAlign.Pos || end > lastAlign.Changes {
		return nil
	}

	for _, a := range n.normalizedString.Alignments {
		if a.Pos >= start && a.Changes <= end {
			selectedAlignments = append(selectedAlignments, a)
		}
	}

	pos := selectedAlignments[0].Pos
	changes := selectedAlignments[len(selectedAlignments)].Changes

	return utils.MakeRange(pos, changes)

}

func (n *Normalized) RangeOf(s string, r []int) (string, error) {
	start := r[0]
	end := r[len(r)]

	if start >= len(s) || end > len(s) {

		err := errors.New("Input range is out of bounds.")
		return "", err
	}

	return s[start:end], nil

}

// Range returns a range of the normalized string (indexing on character not byte)
func (n *Normalized) Range(r []int) (string, error) {
	return n.RangeOf(n.normalizedString.Normalized, r)
}

func (n *Normalized) RangeOriginal(r []int) (string, error) {
	return n.RangeOf(n.normalizedString.Original, r)
}

type ChangeMap struct {
	RuneVal string
	Changes int
}

// Transform applies transformations to the current normalized version, updating the current
// alignments with the new ones.
// This method expect an Iterator yielding each rune of the new normalized string
// with a `change` interger size equals to:
//   - `1` if this is a new rune
//   - `-N` if the char is right before N removed runes
//   - `0` if this rune represents the old one (even if changed)
// Since it is possible that the normalized string doesn't include some of the `characters` (runes) at
// the beginning of the original one, we need an `initial_offset` which represents the number
// of removed runes at the very beginning.
//
// `change` should never be more than `1`. If multiple runes are added, each of
// them has a `change` of `1`, but more doesn't make any sense.
// We treat any value above `1` as `1`.
func (n *Normalized) Transform(m []ChangeMap, initialOffset int) {
	offset := 0
	remainingOffset := initialOffset
	var (
		runeVals  []string
		newAligns []Alignment
	)

	// E.g. string `élégant`
	// Before NFD():  [{233 0} {108 1} {233 2} {103 3} {97 4} {110 5} {116 6}]
	// After NFD(): 	[{101 0} {769 1} {108 2} {101 3} {769 4} {103 5} {97 6} {110 7} {116 8}]
	// New Alignments:
	// {0, 1},
	// {0, 1},
	// {1, 2},
	// {2, 3},
	// {2, 3},
	// {3, 4},
	// {4, 5},
	// {5, 6},
	// {6, 7},

	for i, item := range m {
		var changes int

		if remainingOffset != 0 {
			changes = item.Changes - remainingOffset
			remainingOffset = 0
		} else {
			changes = item.Changes
		}

		// NOTE: offset can be negative or positive value
		// A positive offset means we added `characters` (runes).
		// So we need to remove this offset from the current index to find out the previous id.
		idx := i - offset

		var align Alignment

		switch c := changes; { // Recall: Using switch with no condition. Ref: https://yourbasic.org/golang/switch-statement/
		// Newly added `character`
		case c > 0:
			offset += 1
			if idx < 1 {
				align = Alignment{
					Pos:     0,
					Changes: 0,
				}
			}
			// Get alignment from previous index
			align = n.normalizedString.Alignments[idx-1]

		// No changes
		case c == 0:
			align = n.normalizedString.Alignments[idx]

		// Some `characters` were removed. We merge our range with one from the
		// removed `characters` as the new alignment
		case c < 0:
			var uch = -changes
			offset += changes
			aligns := n.normalizedString.Alignments[idx:(idx + uch)]

			// Find max, min from this slice
			// TODO: improve algorithm? gonum?
			var (
				min int = 0
				max int = 0
			)
			for _, a := range aligns {
				if a.Changes < min {
					min = a.Changes
				}
				if a.Pos < min {
					min = a.Pos
				}

				if max < a.Changes {
					max = a.Changes
				}

				if max < a.Pos {
					max = a.Pos
				}
			}

			align = Alignment{
				Pos:     min,
				Changes: max,
			}
		} // end of Switch block

		newAligns = append(newAligns, align)
		runeVals = append(runeVals, item.RuneVal)

	} // end of For-Range block

	n.normalizedString.Alignments = newAligns
	n.normalizedString.Normalized = strings.Join(runeVals, "")

}

func (n *Normalized) NFD() {

	var changeMap []ChangeMap

	// Create slice of (char, changes) to map changing
	// if added (inserted) rune, changes = 1; `-N` if char
	// right before N removed chars
	// changes = 0 if this represents the old one (even if changed)

	// Iterating over string and apply tranformer (NFD). One character at a time
	// A `character` is defined as:
	// - a sequence of runes that starts with a starter,
	// - a rune that does not modify or combine backwards with any other rune,
	// - followed by possibly empty sequence of non-starters, that is, runes that do (typically accents).
	// We will iterate over string and apply transformer to each char
	// If a char composes of one rune, there no changes
	// If more than one rune, first is no change, the rest is 1 changes
	var it norm.Iter
	it.InitString(norm.NFD, n.normalizedString.Normalized)
	for !it.Done() {
		runes := []rune(string(it.Next()))

		for i, r := range runes {

			switch i := i; {
			case i == 0:
				changeMap = append(changeMap, ChangeMap{
					RuneVal: fmt.Sprintf("%+q", r),
					Changes: 0,
				})
			case i > 0:
				changeMap = append(changeMap, ChangeMap{
					RuneVal: fmt.Sprintf("%+q", r),
					Changes: 1,
				})
			}
		}

	}

	n.Transform(changeMap, 0)

}

// TODO: NFC
// TODO: NFKD
// TODO: NFKC
// TODO: Filter

func (n *Normalized) RemoveAccents() {

	s := n.normalizedString.Normalized
	b := make([]byte, len([]rune(s)))

	tf := transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)

	// TODO: use Tranform and implement chaing to Alignments
	_, _, err := tf.Transform(b, []byte(s), true)
	if err != nil {
		log.Fatal(err)
	}

	n.normalizedString.Normalized = string(b)

}
