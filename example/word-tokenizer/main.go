package main

import (
	"fmt"

	"github.com/sugarme/sermo/tokenize"
	"github.com/sugarme/sermo/vocab"
)

func main() {

	options := []tokenize.Option{
		tokenize.WithWordpieceTokenizer(50, tokenize.DefaultUnknownToken, vocab.DefaultWordpieceVocab),
	}

	fmt.Println("Testing wordpiece tokenizer...")
	fmt.Println(options)

}
