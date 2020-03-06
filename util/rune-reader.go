package util

import (
	"io"
)

// Package `io` provide a `io.RuneReader` interface
// type RuneReader interface{
// 		ReadRune() (r rune, p int, err error)
//  }
//
// The following RuneReader struct implements `io.RuneReader` interface
// So that it can be used as `io.RuneReader` in various use cases such
// as in `regexp` package where `FindReaderIndex` requires `io.RuneReader`
// (https://golang.org/pkg/regexp/#Regexp.FindReaderIndex)

// RuneReader struct implements `io.RuneReader` interface
//
// For example:
// func main() {
// s := "Hello, 世界! 世 界 World 世界 World!"
// rs := []rune(s)
// rreader := NewRuneReader(rs)
//
// re := regexp.MustCompile(`(?i)(\S+界 W\w+)`)
// m := re.FindReaderIndex(&rreader)
//
// fmt.Println(m, string(rs[m[0]:m[1]]))
// }

type RuneReader struct {
	src []rune
	pos int
}

// NewRuneReader create a new of type `io.RuneReader`
func NewRuneReader(r []rune) RuneReader {
	return RuneReader{
		src: r,
	}
}

// ReadRune implements `io.RuneReader` for RuneReader struct
func (r *RuneReader) ReadRune() (rune, int, error) {
	if r.pos >= len(r.src) {
		return -1, 0, io.EOF
	}

	nextRune := r.src[r.pos]
	r.pos++

	return nextRune, 1, nil
}
