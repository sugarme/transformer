// utils slice manupulation
// Ref. https://github.com/golang/go/wiki/SliceTricks
// TODO: function chaining as in : https://github.com/elliotchance/pie
package util

import (
	"errors"
	"math/rand"
	"sort"
)

// CopyStr copy a string slice to another
func CopyStr(a []string) []string {
	var b []string
	b = append(a[:0:0], a...) // See https://github.com/go101/go101/wiki
	return b
}

// CutStr removes a sub-slice from start(inclusive in removed slice)
// to end (exclusive in remve slice) from original slice
func CutStr(a []string, start int, end int) ([]string, error) {
	var err error
	if start < 0 || start > len(a) {
		err = errors.New("`start` is out of bound")
		return nil, err
	}

	if end < 0 || end > len(a) {
		err = errors.New("`end` is out of bound")
		return nil, err
	}

	if start > end {
		err = errors.New("`end` should be greater than `start`")
		return nil, err
	}

	return append(a[:start], a[end:]...), nil
}

// DeleteStr deletes an string item at specified index and preserves order
func DeleteStr(a []string, i int) ([]string, error) {
	var err error
	if i < 0 || i > len(a) {
		err = errors.New("`i` index is out of bound.")
		return nil, err
	}

	return append(a[:i], a[i+1:]...), nil
}

// ExpandStr expands capacity of slice from `start` (inclusive) to `end` exclusive point.
func ExpandStr(a []string, start int, end int) ([]string, error) {
	var err error
	if start < 0 || start > len(a) {
		err = errors.New("`start` is out of bound")
		return nil, err
	}

	if end < 0 || end > len(a) {
		err = errors.New("`end` is out of bound")
		return nil, err
	}

	if start > end {
		err = errors.New("`end` should be greater than `start`")
		return nil, err
	}

	return append(a[:start], append(make([]string, end), a[start:]...)...), nil
}

// ExtendStr extends slice capacity by adding a spcified size at then end of slice.
func ExtendStr(a []string, size int) ([]string, error) {
	if size <= 0 {
		err := errors.New("Extending size should be greater than zero.")
		return nil, err
	}

	return append(a, make([]string, size)...), nil
}

// FilterStr filters in place a string slice
func FilterStr(a []string, ffunc func(string) bool) []string {
	n := 0
	for _, x := range a {
		if ffunc(x) {
			a[n] = x
			n++
		}
	}
	return a[:n]
}

// InsertStr inserts an string to slice of string at specified index i.
func InsertStr(a []string, item string, i int) ([]string, error) {
	var err error
	if i < 0 || i > len(a) {
		err = errors.New("`i` index is out of bound.")
		return nil, err
	}

	return append(a[:i], append([]string{item}, a[i:]...)...), nil

}

// InsertVecStr inserts a slice (`b`) to origin slice (`a`) at specified index point
func InsertVecStr(a []string, b []string, i int) ([]string, error) {
	var err error

	if i < 0 || i > len(a) {
		err = errors.New("`i` index point is out of bound.")
		return nil, err
	}

	return append(a[:i], append(b, a[i:]...)...), nil
}

// PushStr pushes an item to a slice at the end of it.
func PushStr(a []string, item string) []string {
	return append(a, item)
}

// PopStr pops last item out of a give slice
func PopStr(a []string) (string, []string) {
	return a[len(a)-1], a[:len(a)-1]
}

// PushFrontStr pushes an item to the front of a slice (unshift)
func PushFrontStr(a []string, item string) []string {
	return append([]string{item}, a...)
}

// PopFrontStr pops the first item from the slice (shift)
func PopFrontStr(a []string) (string, []string) {
	return a[0], a[1:]
}

// FilterStrNoAllocate filters a slice without allocating.
// This trick uses the fact that a slice shares the same backing array
// and capacity as the original, so the storage is reused for the filtered slice.
// Of course, the original contents are modified.
func FilterStrNoAllocate(a []string, f func(string) bool) []string {
	b := a[:0]
	for _, x := range a {
		if f(x) {
			b = append(b, x)
		}
	}

	// Garbage collected
	for i := len(b); i < len(a); i++ {
		a[i] = "" // nil or the zero value of T
	}

	return b
}

// ReverseStr replaces the contents of a slice with
// the same elements but in reverse order
func ReverseStr(a []string) []string {
	for i := len(a)/2 - 1; i >= 0; i-- {
		opp := len(a) - 1 - i
		a[i], a[opp] = a[opp], a[i]
	}

	return a
}

// ReverseLRStr does the same as ReverseStr except with 2 indices
func ReverseLRStr(a []string) []string {
	for left, right := 0, len(a)-1; left < right; left, right = left+1, right-1 {
		a[left], a[right] = a[right], a[left]
	}

	return a
}

// ShuffleStr shuffles a slice using Fisher–Yates algorithm
// Since go1.10, this is available at math/rand.Shuffle (https://godoc.org/math/rand#Shuffle)
func ShuffleStr(a []string) []string {
	for i := len(a) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}

	return a
}

// BatchStr slits a slice to batches (slice of slices)
func BatchStr(a []string, size int) ([][]string, error) {
	var err error
	if size < 0 {
		err = errors.New("Batch size should be greater than zero")
		return nil, err
	}

	batches := make([][]string, 0, (len(a)+size-1)/size)

	for size < len(a) {
		a, batches = a[size:], append(batches, a[0:size:size])
	}
	batches = append(batches, a)

	return batches, nil
}

// DeduplicateStr de-duplicates a slice in place
func DeduplicateStr(a []string) []string {
	sort.Strings(a)
	j := 0
	for i := 1; i < len(a); i++ {
		if a[j] == a[i] {
			continue
		}
		j++
		// preserve the original data
		// a[i], a[j] = a[j], a[i]
		// only set what is required
		a[j] = a[i]
	}
	result := a[:j+1]
	return result
}
