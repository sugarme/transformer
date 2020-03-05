// utils slice manupulation
// Ref. https://github.com/golang/go/wiki/SliceTricks
package utils

import (
	"errors"
	"reflect"
)

type Slice []interface{}

func (s *Slice) Insert(i uint32, item interface{}) error {
	a := *s
	// Make sure same type
	if len(a) > 0 && reflect.TypeOf(item) != reflect.TypeOf(a[0]) {
		err := errors.New("Element to be inserted is in different type.")
		return err
	}

	a = append(a[:i], append([]interface{}{item}, a[i:]...)...)

	s = &a

	return nil
}
