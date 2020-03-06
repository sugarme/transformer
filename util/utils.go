package util

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/rivo/uniseg"
)

// Makerange creates a sequence of number (range)
// Ref. https://stackoverflow.com/questions/39868029
func MakeRange(min, max int) []int {
	a := make([]int, max-min+1)
	for i := range a {
		a[i] = min + i
	}
	return a
}

// StringIndex returns index (start) for substring on a given string
// It returns -1 and error if not matching
func StringIndex(s, sub string) (index int, err error) {
	i := strings.Index(s, sub)
	if i <= -1 {
		err := errors.New("Index not found")
		return -1, err
	}
	return i, nil
}

// ToASCII converts string to ASCII form
// Ref. https://stackoverflow.com/questions/12668681
func ToASCII(s string) string {
	var as []string
	for _, r := range []rune(s) {
		quoted := strconv.QuoteRuneToASCII(r)
		as = append(as, quoted[1:len(quoted)-1])
	}

	return strings.Join(as, "")
}

// ToGrapheme converts string to grapheme
// TODO: should we include this func?
// Ref: https://github.com/golang/go/issues/14820
func ToGrapheme(s string) string {
	gr := uniseg.NewGraphemes(s)

	var str []string

	for gr.Next() {
		s := fmt.Sprintf("%x", gr.Runes())
		str = append(str, s)
	}

	return strings.Join(str, "")
}

// Ref. https://stackoverflow.com/questions/14000534
type RuneGen func() rune

// MapRune maps ...
func MapRune(g RuneGen, f func(rune) rune) RuneGen {
	return func() rune {
		return f(g())
	}
}

// MinMax returns min and max from input int array
func MinMax(array []int) (int, int) {
	var max int = array[0]
	var min int = array[0]
	for _, value := range array {
		if max < value {
			max = value
		}
		if min > value {
			min = value
		}
	}
	return min, max
}

// StringInSlice check whether given string is in a slice
func StringInSlice(a string, list []string) bool {
	for _, b := range list {
		if b == a {
			return true
		}
	}
	return false
}

// FileSize returns length of given file using `os.Stat()`
// Ref. https://stackoverflow.com/questions/17133590
func FileSize(filepath string) (int64, error) {
	fi, err := os.Stat(filepath)
	if err != nil {
		return 0, err
	}
	// get the size
	return fi.Size(), nil
}

// ReadAllLn reads all line by line from a file using bufio.scanner
func ReadAllLn(filepath string, keepBreakLine bool) ([]string, error) {
	var lines []string
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		l := scanner.Text()
		if keepBreakLine {
			lines = append(lines, fmt.Sprintf("%v\n", l))
		}
		lines = append(lines, l)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return lines, nil
}
