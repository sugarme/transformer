package normalizer

import (
	"errors"
	// "fmt"
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

// Alignment map normalized string to original one
// Pos: the position in the modified string and
// Changes: representing the number (size) of insertions or deletions
type Alignment struct {
	Pos     int
	Changes int
}

// Normalized is wrapper for a `NormalizedString` and provides
// methods to access it.
type Normalized struct {
	normalizedString NormalizedString
}

// func (n Normalized) NewFrom(s string) NormalizedString {
func NewNormalizedFrom(s string) Normalized {
	var alignments []Alignment
	chars := strings.Split(s, "")
	for i, _ := range chars {
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

type RuneItem struct {
	Rune rune
	Pos  int
}

// Transform applies transformations to the current normalized version, updating the current
// alignments with the new ones.
// This method expect an Iterator yielding each char of the new normalized string
// with a `change` interger size equals to:
//   - `1` if this is a new char
//   - `-N` if the char is right before N removed chars
//   - `0` if this char represents the old one (even if changed)
// Since it is possible that the normalized string doesn't include some of the characters at
// the beginning of the original one, we need an `initial_offset` which represents the number
// of removed chars at the very beginning.
//
// `change` should never be more than `1`. If multiple chars are added, each of
// them has a `change` of `1`, but more doesn't make any sense.
// We treat any value above `1` as `1`.
func (n *Normalized) Transform(runeItems []RuneItem, initialOffset int) {
	offset := 0
	remainingOffset := initialOffset
	var (
		newNormalizedRunes []rune
		newAligns          []Alignment
	)

	for i, item := range runeItems {
		var changes int

		if remainingOffset != 0 {
			changes = item.Pos - remainingOffset
			remainingOffset = 0
		} else {
			changes = item.Pos
		}

		var uof int
		if offset < 0 {
			uof = -offset
		} else {
			uof = offset
		}

		// A positive offset means we added `characters`. So we need to remove this offset from
		// the current index to find out the previous id.
		var idx int

		switch os := offset; {
		case os < 0:
			idx = i + uof
		case os >= 0:
			idx = i + uof
		}

		var align Alignment

		switch c := changes; {
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
		newNormalizedRunes = append(newNormalizedRunes, item.Rune)

	} // end of For-Range block

	n.normalizedString.Alignments = newAligns
	n.normalizedString.Normalized = string(newNormalizedRunes)

}

func (n *Normalized) NFD() {
	tf := transform.Chain(norm.NFD)
	newNormalized, _, err := transform.String(tf, n.normalizedString.Normalized)
	if err != nil {
		log.Fatal(err)
	}

	var items []RuneItem

	for i, r := range newNormalized {
		items = append(items, RuneItem{Rune: r, Pos: i})
	}

}

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
