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

	fmt.Println(model.GetVocabSize())

	tk := tokenizer.NewTokenizer(model)

	fmt.Printf("Vocab size: %v\n", tk.GetVocabSize(false))

	bl := pretokenizer.NewByteLevel()

	tk.WithPreTokenizer(bl)

	en := tk.Encode("Mi estas Julien.")

	fmt.Printf("Encoding: %v\n", en)

	for _, tok := range en.GetTokens() {
		fmt.Println(tok)
	}

}
