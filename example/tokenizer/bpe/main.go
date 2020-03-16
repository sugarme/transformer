package main

import (
	"fmt"
	"log"

	"github.com/sugarme/sermo/tokenizer"
	bpe "github.com/sugarme/sermo/tokenizer/model/bpe"
)

func main() {

	files := []string{
		"./data-sample.txt",
	}

	model, err := bpe.NewBPE()
	if err != nil {
		log.Fatal(err)
	}

	trainer := bpe.NewBpeTrainer(2, 50)

	tk := tokenizer.NewTokenizer(model)

	err = tk.Train(trainer, files)
	if err != nil {
		log.Fatal(err)
	}

	// Print out some data
	// vocab := model.GetVocab()
	trainedModel := tk.GetModel()
	vocab := trainedModel.(*bpe.BPE).Vocab
	merges := trainedModel.(*bpe.BPE).Merges

	fmt.Println(vocab)
	fmt.Println(*merges)
}
