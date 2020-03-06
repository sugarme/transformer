package util

// RuneIter is rune iterator with Next() method.
// Ref. https://stackoverflow.com/questions/14000534
type RuneIter struct {
	Index int
}

func (r *RuneIter) Next() rune {
	r.Index += 1
	return rune(r.Index)
}
