package main

import (
	"fmt"
	"log"
	// "runtime"

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
	device := gotch.CPU
	// device := gotch.NewCuda().CudaIfAvailable()
	vs := nn.NewVarStore(device)

	lr := 1e-4
	opt, err := nn.DefaultAdamConfig().Build(vs, lr)
	if err != nil {
		log.Fatal(err)
	}

	_ = debug.UsedCPUMem()
	model := bert.NewBertForQuestionAnswering(vs.Root(), config)
	err = vs.Load("../../data/bert/bert-qa.ot")
	// _, err = vs.LoadPartial("./bert-qa-squad-ck000005.gt")
	if err != nil {
		log.Fatalf("Load model weight error: \n%v", err)
	}

	// err = saveCheckPoint(vs, "test.gt")
	// if err != nil {
	// log.Fatal(err)
	// }

	debug.UsedCPUMem()

	var batchSize int64 = 1
	var seqLen int64 = int64(384)
	batches := int(dataset.MustSize()[1])/int(batchSize) - 1
	checkPoint := 5

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

		// ts.MustGradSetEnabled(true)
		startLogits, _, _, _, err := model.ForwardT(inputIds, ts.NewTensor(), typeIds, ts.NewTensor(), ts.NewTensor(), true)
		if err != nil {
			log.Fatal(err)
		}

		startLoss := startLogits.CrossEntropyForLogits(startA)
		opt.BackwardStep(startLoss)

		// ts.MustGradSetEnabled(false)

		loss := startLoss.Float64Values()[0]
		// startLoss.MustDrop()

		startLogits.MustDrop()
		// endLogits.MustDrop()

		/* for i := 0; i < len(allAttentionMasks); i++ {
		 *   allAttentionMasks[i].MustDrop()
		 * }
		 * for i := 0; i < len(allAttentions); i++ {
		 *   allAttentions[i].MustDrop()
		 * } */

		// typeIds.MustDrop()
		// inputIds.MustDrop()
		// startA.MustDrop()

		// runtime.GC()
		fmt.Printf("Batch %3.0d\tLoss: %8.3f\tUsed GPU: %8.0f \tUsed RAM: %8.0f\n", n, loss, debug.UsedGPUMem(), debug.UsedCPUMem())
		// fmt.Printf("Batch %3.0d\tUsed GPU: %8.0f \tUsed RAM: %8.0f\n", n, debug.UsedGPUMem(), debug.UsedCPUMem())
		// })

		// Save model at check point
		if n > 0 && n%checkPoint == 0 {
			fmt.Printf("Saving model at checkpoint %06d...\n", currIdx)
			filepath := fmt.Sprintf("bert-qa-squad-ck%06d.gt", currIdx)
			err := vs.Save(filepath)
			if err != nil {
				log.Println(err)
			}
		}

		// next batch
		currIdx = nextIdx
		nextIdx += batchSize
	}
}

func toInt64(data []int) []int64 {
	var data64 []int64
	for _, v := range data {
		data64 = append(data64, int64(v))
	}

	return data64
}

func saveCheckPoint(vs nn.VarStore, filePath string) error {
	// runtime.GC()
	// var namedTensors []ts.NamedTensor
	// for k, v := range vs.Vars.NamedVariables {
	// namedTensors = append(namedTensors, ts.NamedTensor{
	// Name:   k,
	// Tensor: v,
	// })
	// }
	//
	// return ts.SaveMulti(namedTensors, filePath)

	return vs.Save(filePath)
}
