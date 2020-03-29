package main

import (
	"fmt"
	"log"

	"github.com/sugarme/sermo/pretokenizer"
	"github.com/sugarme/sermo/tokenizer"
	bpe "github.com/sugarme/sermo/tokenizer/model/bpe"
)

func main() {
	model, err := bpe.NewBpeFromFiles("example/tokenizer/bpe/test/model/es-vocab.json", "example/tokenizer/bpe/test/model/es-merges.txt")
	if err != nil {
		log.Fatal(err)
	}

	tk := tokenizer.NewTokenizer(model)

	bl := pretokenizer.NewByteLevel()

	tk.WithPreTokenizer(bl)

	en := tk.Encode("Mi estas Julien.")

	for _, tok := range en.GetTokens() {
		fmt.Printf("'%v'\n", tok)
	}

}
