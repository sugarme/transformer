package main

import (
	"fmt"
	// "log"

	// "golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

type ChangeMap struct {
	RuneVal string
	Changes int
}

func NFD(s string) (m []ChangeMap) {

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
	it.InitString(norm.NFD, s)
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

	return changeMap

}

func main() {
	s := "élégant"
	m := NFD(s)
	fmt.Println(m)

}
