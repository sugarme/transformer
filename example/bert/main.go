package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	// "github.com/sugarme/tokenizer"
	"github.com/sugarme/transformer/bert"
)

func main() {

	device := gotch.CPU
	vs := nn.NewVarStore(device)

	// bertTokenizer := tokenizer.BertTokenizerFromFile("../../data/bert/vocab.txt")
	config := bert.ConfigFromFile("../../data/bert/config.json")
	fmt.Printf("Bert Configuration:\n%+v\n", config)

	model := bert.NewBertForMaskedLM(vs.Root(), config)
	err := vs.Load("../../data/bert/model.ot")
	if err != nil {
		log.Fatalf("Load model weight error: \n%v", err)
	}

	fmt.Printf("Varstore weights have been loaded\n")
	fmt.Printf("Num of variables: %v\n", len(vs.Variables()))

	// fmt.Printf("%v\n", vs.Variables())

	fmt.Printf("Bert is Decoder: %v\n", model.Bert.IsDecoder)
}
