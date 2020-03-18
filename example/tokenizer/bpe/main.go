package main

import (
	// "fmt"
	"log"

	"github.com/sugarme/sermo/tokenizer"
	bpe "github.com/sugarme/sermo/tokenizer/model/bpe"
)

func main() {

	files := []string{
		"example/tokenizer/bpe/data-sample.txt",
		"example/tokenizer/bpe/oscar.eo.txt",
	}

	model, err := bpe.NewBPE()
	if err != nil {
		log.Fatal(err)
	}

	trainer := bpe.NewBpeTrainer(2, 52000)

	tk := tokenizer.NewTokenizer(model)

	tk.AddSpecialTokens([]string{
		"<s>",
		"<pad>",
		"</s>",
		"<unk>",
		"<mask>",
	})

	err = tk.Train(trainer, files)
	if err != nil {
		log.Fatal(err)
	}

	// Print out some data
	// vocab := model.GetVocab()
	trainedModel := tk.GetModel()
	// vocab := trainedModel.(*bpe.BPE).Vocab
	// fmt.Println(vocab)

	// merges := trainedModel.(*bpe.BPE).Merges
	// fmt.Println(*merges)

	trainedModel.Save("example/tokenizer/bpe", "es")

}
