package main

import (
	"fmt"
)

func main() {
	dataset := LoadSquadV2("dev")
	fmt.Println("SQuAD data is loaded.")
	fmt.Printf("Dataset shape: %v\n", dataset.MustSize())

	runTrain(dataset)
	// runTrainFromCheckPoint(dataset, "./bert-qa-squad-ck000013.gt", 14)
	// runTrainFromScratch(dataset) // NOTE. need to modify config file to add more layers.
}
