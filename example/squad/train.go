package main

import (
	// "fmt"
	"log"
	"runtime"

	"github.com/sugarme/transformer/bert"
	"github.com/sugarme/transformer/util/debug"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func runTrain(dataset ts.Tensor) {
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

	var batchSize int64 = 4
	var seqLen int64 = int64(384)
	batches := 20

	var currIdx int64 = 0
	var nextIdx int64 = batchSize
	for n := 0; n < batches; n++ {
		ts.NoGrad(func() {
			inputIdsIdx := []ts.TensorIndexer{ts.NewSelect(0), ts.NewNarrow(currIdx, nextIdx)}
			inputIds := dataset.Idx(inputIdsIdx).MustTo(device, true).MustView([]int64{batchSize, seqLen}, true)

			typeIdsIdx := []ts.TensorIndexer{ts.NewSelect(1), ts.NewNarrow(currIdx, nextIdx)}
			typeIds := dataset.Idx(typeIdsIdx).MustTo(device, true).MustView([]int64{batchSize, seqLen}, true)

			startLogits, endLogits, allAttentionMasks, allAttentions, err := model.ForwardT(inputIds, ts.None, typeIds, ts.None, ts.None, true)
			if err != nil {
				log.Fatal(err)
			}
			startLogits.MustDrop()
			endLogits.MustDrop()

			for i := 0; i < len(allAttentionMasks); i++ {
				allAttentionMasks[i].MustDrop()
				allAttentions[i].MustDrop()
			}

			inputIds.MustDrop()
			typeIds.MustDrop()

			runtime.GC()
		})

		// next batch
		currIdx = nextIdx
		nextIdx += batchSize

		debug.UsedRAM()
		// debug.UsedGPU()
		// fmt.Printf("Batch %v completed.\n", n)
	}
}

func toInt64(data []int) []int64 {
	var data64 []int64
	for _, v := range data {
		data64 = append(data64, int64(v))
	}

	return data64
}
