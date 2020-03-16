package util

import (
	"errors"
	"fmt"
	"log"
	"runtime"
	"strings"
)

// ErrorContains checks if the error message in out contains the text in want
// Ref. https://stackoverflow.com/questions/42035104
//
// Example usage
// if !ErrorContains(err, "unexpected banana") {
// t.Errorf("unexpected error: %v", err)
// }
//
// if !ErrorContains(err, "") {
// t.Errorf("unexpected error: %v", err)
// }
func ErrorContains(out error, want string) bool {
	if out == nil {
		return want == ""
	}
	if want == "" {
		return false
	}
	return strings.Contains(out.Error(), want)
}

// HandleError wraps original error message with function
// and source code position where error is captured.
// Ref. https://stackoverflow.com/questions/24809287
func TraceError(err error) error {
	if err != nil {
		// notice that we're using 1, so it will actually log the where
		// the error happened, 0 = this function, we don't want that.
		pc, fn, line, _ := runtime.Caller(1)

		msg := fmt.Sprintf("[error] in %s \n[%s:%d] \n%v", runtime.FuncForPC(pc).Name(), fn, line, err)

		newErr := errors.New(msg)
		return newErr

	}

	return nil
}

// LogError logs error with the function name as well.
func LogError(err error) {
	if err != nil {
		// notice that we're using 1, so it will actually log the where
		// the error happened, 0 = this function, we don't want that.
		pc, fn, line, _ := runtime.Caller(1)

		log.Printf("[error] in %s \n[%s:%d] \n%v", runtime.FuncForPC(pc).Name(), fn, line, err)
	}
}
