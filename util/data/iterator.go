package data

// Iterator defines `Next()` method with two
// return values (data value and whether it is valid)
type Iterator interface {
	Next() (value interface{}, ok bool)
}
