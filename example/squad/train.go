package main

import (
	"fmt"
	"log"
	"runtime"

	"github.com/sugarme/transformer/bert"
	"github.com/sugarme/transformer/util/debug"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func runTrain(dataset ts.Tensor) {
	// runtime.GOMAXPROCS(4)

	// Config
	config, err := bert.ConfigFromFile("../../data/bert/config-qa.json")
	if err != nil {
		log.Fatal(err)
	}

	// Model
	// device := gotch.CPU
	device := gotch.NewCuda().CudaIfAvailable()
	vs := nn.NewVarStore(device)

	lr := 1e-4
	opt, err := nn.DefaultAdamConfig().Build(vs, lr)
	if err != nil {
		log.Fatal(err)
	}

	_ = debug.UsedCPUMem()
	model := bert.NewBertForQuestionAnswering(vs.Root(), config)
	err = vs.Load("../../data/bert/bert-qa.ot")
	if err != nil {
		log.Fatalf("Load model weight error: \n%v", err)
	}

	debug.UsedCPUMem()

	var batchSize int64 = 4
	var seqLen int64 = int64(384)
	batches := 1000

	var currIdx int64 = 0
	var nextIdx int64 = batchSize
	for n := 0; n < batches; n++ {
		// ts.NoGrad(func() {
		inputIdsIdx := []ts.TensorIndexer{ts.NewSelect(0), ts.NewNarrow(currIdx, nextIdx)}
		inputIds := dataset.Idx(inputIdsIdx).MustView([]int64{batchSize, seqLen}, true).MustTo(device, true)
		// inputIds := ts.MustZeros([]int64{batchSize, seqLen}, gotch.Int64, device)

		typeIdsIdx := []ts.TensorIndexer{ts.NewSelect(1), ts.NewNarrow(currIdx, nextIdx)}
		typeIds := dataset.Idx(typeIdsIdx).MustView([]int64{batchSize, seqLen}, true).MustTo(device, true)
		// typeIds := ts.MustZeros([]int64{batchSize, seqLen}, gotch.Int64, device)

		startIdx := []ts.TensorIndexer{ts.NewSelect(3), ts.NewNarrow(currIdx, nextIdx), ts.NewNarrow(0, 1)}
		startA := dataset.Idx(startIdx).MustView([]int64{batchSize}, true).MustTo(device, true)
		// startA := ts.MustOnes([]int64{batchSize}, gotch.Int64, device)

		ts.MustGradSetEnabled(true)
		startLogits, endLogits, allAttentionMasks, allAttentions, err := model.ForwardT(inputIds, ts.NewTensor(), typeIds, ts.NewTensor(), ts.NewTensor(), true)
		if err != nil {
			log.Fatal(err)
		}

		endLogits.MustDrop()

		startLoss := startLogits.CrossEntropyForLogits(startA)
		opt.BackwardStep(startLoss)

		// ts.MustGradSetEnabled(false)

		loss := startLoss.Float64Values()[0]

		startLoss.MustDrop()

		startLogits.MustDrop()

		for i := 0; i < len(allAttentionMasks); i++ {
			allAttentionMasks[i].MustDrop()
			allAttentions[i].MustDrop()
		}

		inputIds.MustDrop()
		typeIds.MustDrop()

		runtime.GC()
		fmt.Printf("Batch %3.0d\tLoss: %8.3f\tUsed GPU: %8.0f\n", n, loss, debug.UsedGPUMem())
		// })

		// next batch
		currIdx = nextIdx
		nextIdx += batchSize

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
