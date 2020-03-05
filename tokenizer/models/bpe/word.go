package bpe

type Merge struct {
	Pos   uint
	Rank  uint32
	NewId uint32
}

// Ordering is a enum of Less, Equal, and Greater
type Ordering int

const (
	Less    Ordering = -1
	Equal   Ordering = 0
	Greater Ordering = 1
)

// NOTE.Should  we implement comparing methods?
// - Eq
// - PartialCmp
// - Cmp
func (m *Merge) Eq(other *Merge) bool {
	return m.Rank == other.Rank && m.Pos == other.Pos
}

func (m *Merge) PartialCmp(other *Merge) (Ordering, error) {
	// First, compare rank
	if m.Rank != other.Rank {
		if other.Rank < m.Rank {
			return Less, nil
		} else if other.Rank > m.Rank {
			return Greater, nil
		}
	}
	// Then, compare pos
	if other.Pos < m.Pos {
		return Less, nil
	} else if other.Pos > m.Pos {
		return Greater, nil
	} else {
		return Equal, nil
	}
}

func (m *Merge) Cmp(other *Merge) Ordering {
	res, _ := m.PartialCmp(other)
	return res
}

type Symbol struct {
	C    *uint32
	Prev *int
	Next *int
	Len  *uint
}

func (s *Symbol) MergeWith(other *Symbol, newC uint32) {
	s.C = &newC
	*s.Len += *other.Len
	s.Next = other.Next
}

type Word struct {
	Symbols []Symbol
}

func NewWord() Word {
	return Word{
		Symbols: []Symbol{},
	}
}

func (w *Word) Add(c uint32) {
	var (
		prev, next int
	)
	len := len(w.Symbols)
	last := w.Symbols[len]

	if last.Next != nil {
		// update `next` on previous one
		w.Symbols[len].Next = &len
		prev = len - 1
		next = -1
	} else {
		prev = -1
		next = -1
	}

	var sLen uint = 1 // NOTE: assign 1 to a variable so that we can take address of it.

	w.Symbols = append(w.Symbols, Symbol{
		C:    &c,
		Prev: &prev,
		Next: &next,
		Len:  &sLen,
	})
}

type Pair struct {
	C1 *uint32
	C2 *uint32
}

type WChange struct {
	C1     *uint32
	C2     *uint32
	Change int32
}

func (w *Word) Merge(c1, c2, replacement uint32) []WChange {
	var changes []WChange

	for i := 0; i < len(w.Symbols); i++ {

		// found a pair
		if w.Symbols[i].C == &c1 && (i+1) < len(w.Symbols) && w.Symbols[i+1].C == &c2 {
			first := w.Symbols[i]
			second := w.Symbols[i+1]

			// If there's other characters before the pair
			if i > 0 {
				changes = append(changes, WChange{
					C1:     w.Symbols[i-1].C,
					C2:     first.C,
					Change: -1,
				})
				changes = append(changes, WChange{
					C1:     w.Symbols[i-1].C,
					C2:     &replacement,
					Change: 1,
				})
			}

			// Remove in place
			newLen := *first.Len + *second.Len
			newS := Symbol{
				C:    &replacement,
				Prev: first.Prev,
				Next: second.Next,
				Len:  &newLen,
			}

		}
	}

}
