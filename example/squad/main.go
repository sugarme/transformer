package main

import (
	"fmt"
)

func main() {
	dataset := LoadSquadV2("dev")
	fmt.Println("SQuAD data is loaded.")
	fmt.Printf("Dataset shape: %v\n", dataset.MustSize())
	runTrain(dataset)
	// runTrainFromScratch(dataset)
}
