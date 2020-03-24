package main

import (
	"fmt"
	"log"
	"time"

	"github.com/sugarme/sermo/pretokenizer"
	"github.com/sugarme/sermo/tokenizer"
	bpe "github.com/sugarme/sermo/tokenizer/model/bpe"
)

func main() {

	startTime := time.Now()

	files := []string{
		// "example/tokenizer/bpe/input/data-sample.txt",
		"example/tokenizer/bpe/input/oscar.eo.txt",
		// "example/tokenizer/bpe/input/trainer-sample.txt",
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

	bytelevel := pretokenizer.NewByteLevel()

	tk.WithPreTokenizer(bytelevel)

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

	trainedModel.Save("example/tokenizer/bpe/output", "es")

	trainedTime := time.Since(startTime).Seconds() / 60

	fmt.Printf("Training time (min): %f.2\n", trainedTime)

}
