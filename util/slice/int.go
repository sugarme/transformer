// utils slice manupulation
// Ref. https://github.com/golang/go/wiki/SliceTricks
// TODO: function chaining as in : https://github.com/elliotchance/pie
package util

import (
	"errors"
	"math/rand"
	"sort"
)

// CopyInt copy a string slice to another
func CopyInt(a []int) []int {
	var b []int
	b = append(a[:0:0], a...) // See https://github.com/go101/go101/wiki
	return b
}

// CutInt removes a sub-slice from start(inclusive in removed slice)
// to end (exclusive in remve slice) from original slice
func CutInt(a []int, start int, end int) ([]int, error) {
	var err error
	if start < -1 || start > len(a) {
		err = errors.New("`start` is out of bound")
		return nil, err
	}

	if end < -1 || end > len(a) {
		err = errors.New("`end` is out of bound")
		return nil, err
	}

	if start > end {
		err = errors.New("`end` should be greater than `start`")
		return nil, err
	}

	return append(a[:start], a[end:]...), nil
}

// DeleteInt deletes an string item at specified index and preserves order
func DeleteInt(a []int, i int) ([]int, error) {
	var err error
	if i < 0 || i > len(a) {
		err = errors.New("`i` index is out of bound.")
		return nil, err
	}

	return append(a[:i], a[i+1:]...), nil
}

// ExpandInt expands capacity of slice from `start` (inclusive) to `end` exclusive point.
func ExpandInt(a []int, start int, end int) ([]int, error) {
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

	return append(a[:start], append(make([]int, end), a[start:]...)...), nil
}

// ExtendInt extends slice capacity by adding a spcified size at then end of slice.
func ExtendInt(a []int, size int) ([]int, error) {
	if size <= 0 {
		err := errors.New("Extending size should be greater than zero.")
		return nil, err
	}

	return append(a, make([]int, size)...), nil
}

// FilterInt filters in place a int slice
func FilterInt(a []int, ffunc func(int) bool) []int {
	n := 0
	for _, x := range a {
		if ffunc(x) {
			a[n] = x
			n++
		}
	}
	return a[:n]
}

// InsertInt inserts an int to slice of int at specified index i.
func InsertInt(a []int, item int, i int) ([]int, error) {
	var err error
	if i < 0 || i > len(a) {
		err = errors.New("`i` index is out of bound.")
		return nil, err
	}

	return append(a[:i], append([]int{item}, a[i:]...)...), nil

}

// InsertVecInt inserts a slice (`b`) to origin slice (`a`) at specified index point
func InsertVecInt(a []int, b []int, i int) ([]int, error) {
	var err error

	if i < 0 || i > len(a) {
		err = errors.New("`i` index point is out of bound.")
		return nil, err
	}

	return append(a[:i], append(b, a[i:]...)...), nil
}

// PushInt pushes an item to a slice at the end of it.
func PushInt(a []int, item int) []int {
	return append(a, item)
}

// PopInt pops last item out of a give slice
func PopInt(a []int) (int, []int) {
	return a[len(a)-1], a[:len(a)-1]
}

// PushFrontInt pushes an item to the front of a slice (unshift)
func PushFrontInt(a []int, item int) []int {
	return append([]int{item}, a...)
}

// PopFrontInt pops the first item from the slice (shift)
func PopFrontInt(a []int) (int, []int) {
	return a[0], a[1:]
}

// FilterStrNoAllocate filters a slice without allocating.
// This trick uses the fact that a slice shares the same backing array
// and capacity as the original, so the storage is reused for the filtered slice.
// Of course, the original contents are modified.
func FilterIntNoAllocate(a []int, f func(int) bool) []int {
	b := a[:0]
	for _, x := range a {
		if f(x) {
			b = append(b, x)
		}
	}

	// Garbage collected
	for i := len(b); i < len(a); i++ {
		a[i] = 0 // nil or the zero value of T
	}

	return b
}

// ReverseInt replaces the contents of a slice with
// the same elements but in reverse order
func ReverseInt(a []int) []int {
	for i := len(a)/2 - 1; i >= 0; i-- {
		opp := len(a) - 1 - i
		a[i], a[opp] = a[opp], a[i]
	}

	return a
}

// ReverseLRInt does the same as ReverseInt except with 2 indices
func ReverseLRInt(a []int) []int {
	for left, right := 0, len(a)-1; left < right; left, right = left+1, right-1 {
		a[left], a[right] = a[right], a[left]
	}

	return a
}

// ShuffleInt shuffles a slice using Fisher–Yates algorithm
// Since go1.10, this is available at math/rand.Shuffle (https://godoc.org/math/rand#Shuffle)
func ShuffleInt(a []int) []int {
	for i := len(a) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}

	return a
}

// BatchInt slits a slice to batches (slice of slices)
func BatchInt(a []int, size int) ([][]int, error) {
	var err error
	if size < 0 {
		err = errors.New("Batch size should be greater than zero")
		return nil, err
	}

	batches := make([][]int, 0, (len(a)+size-1)/size)

	for size < len(a) {
		a, batches = a[size:], append(batches, a[0:size:size])
	}
	batches = append(batches, a)

	return batches, nil
}

// DeduplicateInt de-duplicates a slice in place
func DeduplicateInt(a []int) []int {
	sort.Ints(a)
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
