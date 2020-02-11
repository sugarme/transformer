// Package vocab provides tools for generating and
// working on dictionary
package vocab

import (
	"fmt"
)

type Token struct {
}

type Dict struct {
	Name   string
	Tokens []Token
}

func NewDict([]Token) Dict {
	fmt.Println("New Dictionary Func...")
	return Dict{}
}
