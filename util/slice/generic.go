package util

import (
	"reflect"
)

// Contain checks whether a element is in a slice (same type)
func Contain(item interface{}, slice interface{}) bool {
	s := reflect.ValueOf(slice)

	if s.Kind() != reflect.Slice {
		panic("Invalid data-type: item and slice are not with the same type.")
	}

	for i := 0; i < s.Len(); i++ {
		if s.Index(i).Interface() == item {
			return true
		}
	}

	return false
}
