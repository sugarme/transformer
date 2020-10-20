package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	"github.com/sugarme/transformer/bert"
)

func main() {
	// Config
	config, err := bert.ConfigFromFile("../../data/bert/config-qa.json")
	if err != nil {
		log.Fatal(err)
	}

	// Model
	device := gotch.CPU
	// device := gotch.NewCuda().CudaIfAvailable()
	vs := nn.NewVarStore(device)

	model := bert.NewBertForQuestionAnswering(vs.Root(), config)
	err = vs.Load("../../data/bert/bert-qa.ot")
	if err != nil {
		log.Fatalf("Load model weight error: \n%v", err)
	}

	// NOTE. BERT finetuned for question answering used different vocab file.
	tk := pretrained.BertLargeCasedWholeWordMaskingSquad()
	maxLen := 512
	truncParams := tokenizer.TruncationParams{
		MaxLength: maxLen,
		Strategy:  tokenizer.OnlySecond,
		Stride:    0,
	}
	tk.WithTruncation(&truncParams)

	// Loop to get <context><question> and infer
	scn := bufio.NewScanner(os.Stdin)
	fmt.Printf("Enter Context (can be multipe lines). Then <ctl+space><enter> to finish:\n")
	var ctxLines []string
	for scn.Scan() {
		line := scn.Text()
		if len(line) == 1 {
			// NULL character (code 0): ctrl-@ (<ctrl+space>)
			if line[0] == '\x00' {
				break
			}
		}
		ctxLines = append(ctxLines, line)
	}
	context := strings.Join(ctxLines, "\n")

	for {

		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Enter question: ")
		question, _ := reader.ReadString('\n')

		// Infering
		encoding, err := tk.EncodePair(question, context, true)
		if err != nil {
			log.Fatal(err)
		}

		var batchSize int64 = 1
		var seqLen int64 = int64(len(encoding.Ids))

		inputTensor := ts.MustOfSlice(toInt64(encoding.Ids)).MustTo(device, true).MustView([]int64{batchSize, seqLen}, true)
		tokenTypeIds := ts.MustOfSlice(toInt64(encoding.TypeIds)).MustTo(device, true).MustView([]int64{batchSize, seqLen}, true)

		var startScores, endScores ts.Tensor
		ts.NoGrad(func() {
			startScores, endScores, _, _, err = model.ForwardT(inputTensor, ts.None, tokenTypeIds, ts.None, ts.None, false)
			if err != nil {
				log.Fatal(err)
			}
		})

		answerStart := startScores.MustGet(0).MustArgmax(0, false, true).Int64Values()[0]
		answerEnd := endScores.MustGet(0).MustArgmax(0, false, true).Int64Values()[0]
		startScores.MustDrop()
		endScores.MustDrop()

		fmt.Printf("answer start: '%v'\n", answerStart)
		fmt.Printf("answer end: '%v'\n", answerEnd)

		// NOTE. To prevent picking `answerStart` > `answerEnd`, we need to find
		// the highest total score for which end >= start
		// TODO.
		// - implement it (pytorch topK) to find topK scores with pair(start, end)
		//   then pick the highest score where start < end
		//   answerTokens := encoding.Tokens[answerStart : answerEnd+1]
		// - Collect all context in multiple chunks and process chunks to get scores
		//   Answer will be the best score.
		answerIds := encoding.Ids[answerStart : answerEnd+1]
		if int(answerEnd) >= len(encoding.Tokens) {
			// answerTokens = encoding.Tokens[answerStart:answerEnd]
			answerIds = encoding.Ids[answerStart:answerEnd]
		}

		answerStr := tk.Decode(answerIds, false)

		// fmt.Printf("context: '%v'\n", strings.Join(encoding.Tokens, " "))
		// fmt.Printf("Answer: '%v'\n", strings.Join(answerTokens, " "))
		fmt.Printf("Context: %v\n", context)
		fmt.Printf("Question: '%v'\n", question)
		fmt.Printf("Answer: '%v'\n", answerStr)

		if err := scn.Err(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			break
		}
		if len(ctxLines) == 0 {
			break
		}
	}
}

func toInt64(data []int) []int64 {
	var data64 []int64
	for _, v := range data {
		data64 = append(data64, int64(v))
	}

	return data64
}

func filterPosition(data []int) []int {
	var filterData []int
	for _, v := range data {
		if v != -1 {
			filterData = append(filterData, v)
		}
	}
	return filterData
}
