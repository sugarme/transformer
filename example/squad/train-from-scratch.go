package main

import (
	"fmt"
	"log"

	"github.com/sugarme/transformer/bert"
	"github.com/sugarme/transformer/util/debug"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func runTrainFromScratch(dataset ts.Tensor) {
	// runtime.GOMAXPROCS(4)

	// Model
	device := gotch.CPU
	// device := gotch.NewCuda().CudaIfAvailable()
	vs := nn.NewVarStore(device)

	// Config
	config, err := bert.ConfigFromFile("../../data/bert/bert-base-uncased-config.json")
	if err != nil {
		log.Fatal(err)
	}
	_ = debug.UsedCPUMem()
	bertBaseUncased := bert.NewBertForMaskedLM(vs.Root(), config)
	err = vs.Load("../../data/bert/bert-base-uncased-model.ot")
	if err != nil {
		log.Fatalf("Load model weight error: \n%v", err)
	}

	model := bert.NewBertForQuestionAnsweringFromBertModel(bertBaseUncased, vs.Root(), config)

	// err = vs.Save("bert-base-uncased-qa-scratch.gt")
	// if err != nil {
	// log.Fatal(err)
	// }

	// Optimizer
	lr := 1e-4
	opt, err := nn.DefaultAdamConfig().Build(vs, lr)
	if err != nil {
		log.Fatal(err)
	}

	debug.UsedCPUMem()

	var batchSize int64 = 1
	var seqLen int64 = int64(384)
	// batches := int(dataset.MustSize()[1])/int(batchSize) - 1
	batches := 100
	checkPoint := 50

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
		startLogits, noop1, _, _, err := model.ForwardT(inputIds, ts.NewTensor(), typeIds, ts.NewTensor(), ts.NewTensor(), true)
		if err != nil {
			log.Fatal(err)
		}

		startLoss := startLogits.CrossEntropyForLogits(startA)
		opt.BackwardStep(startLoss)

		// ts.MustGradSetEnabled(false)

		loss := startLoss.Float64Values()[0]
		startLoss.MustDrop()

		startLogits.MustDrop()
		// endLogits.MustDrop()

		noop1.MustDrop()
		// for i := 0; i < len(noop2); i++ {
		// if noop2[i].MustDefined() {
		// noop2[i].MustDrop()
		// }
		// }
		// for i := 0; i < len(noop3); i++ {
		// if noop3[i].MustDefined() {
		// noop3[i].MustDrop()
		// }
		// }

		// for i := 0; i < len(allAttentionMasks); i++ {
		// allAttentionMasks[i].MustDrop()
		// }

		// for i := 0; i < len(allAttentions); i++ {
		// allAttentions[i].MustDrop()
		// }

		// typeIds.MustDrop()
		// inputIds.MustDrop()
		// startA.MustDrop()

		// runtime.GC()
		fmt.Printf("Batch %3.0d\tLoss: %8.3f\tUsed GPU: %8.0f \tUsed RAM: %8.0f\n", n, loss, debug.UsedGPUMem(), debug.UsedCPUMem())
		// fmt.Printf("Batch %3.0d\tUsed GPU: %8.0f \tUsed RAM: %8.0f\n", n, debug.UsedGPUMem(), debug.UsedCPUMem())
		// })

		/*     // Save model every 20k steps
		 *     if currIdx%20000 == 0 {
		 *       filepath := fmt.Sprintf("bert-qa-finetuned-batches-%v.gt", currIdx)
		 *       err := vs.Save(filepath)
		 *       if err != nil {
		 *         log.Println(err)
		 *       }
		 *     }
		 *  */

		// save model checkpoint
		if n != 0 && n%checkPoint == 0 {
			fmt.Printf("saving checkpoint %06d...\n", currIdx)
			filePath := fmt.Sprintf("bert-qa-squad-ck%06d.gt", currIdx)
			err = vs.Save(filePath)
			if err != nil {
				log.Fatal(err)
			}
		}

		// next batch
		currIdx = nextIdx
		nextIdx += batchSize
	}

}
