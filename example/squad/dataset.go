package main

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"

	"github.com/sugarme/tokenizer/pretrained"
	"github.com/sugarme/transformer/dataset/squad"

	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/tokenizer"
)

func loadSquadV2(datasetName string, isTraining, returnTensorDataset bool) {
	tk := pretrained.BertLargeCasedWholeWordMaskingSquad()
	tk.AddSpecialTokens([]tokenizer.AddedToken{tokenizer.NewAddedToken("[PAD]", true)})

	examples := squad.LoadV2(datasetName)

	// Tensor Size: [6 or 8, Number of features, maxSeqLen]
	features, dataset := squad.ConvertExamplesToFeatures(examples, tk, "bert", 384, 128, 64, "[SEP]", "[PAD]", 0, isTraining, returnTensorDataset)

	fmt.Println("SQuAD V2 is loaded.")
	fmt.Printf("num of features: %v\n", len(features))
	if returnTensorDataset {
		fmt.Printf("dataset shape: %v\n", dataset.MustSize())
	}

	// save feature data
	filePath := fmt.Sprintf("/tmp/squadv2-%v.gob", datasetName)
	err := cacheFeatures(features, filePath)
	if err != nil {
		log.Fatal(err)
	}

	// save dataset tensor
	if returnTensorDataset {
		filePath := fmt.Sprintf("/tmp/squadv2-%v.gt", datasetName)
		dataset.MustSave(filePath)
	}
}

// NOTE. datasetName can be either "dev" or "train", otherwise, panic.
// This is default to BERT and fix params.
// TODO. setup input params
func LoadSquadV2(datasetName string) *ts.Tensor {

	filePath := fmt.Sprintf("/tmp/squadv2-%v.gt", datasetName)

	if isFileNotExist(filePath) {
		fmt.Printf("SQuAD dataset is not cached. Generating and caching...\n")
		loadSquadV2(datasetName, true, true)
	}

	dataset := ts.MustLoad(filePath)
	/*
	 *   // NOTE. this block for debugging only.
	 *   // Feature index
	 *   idsIdx := []ts.TensorIndexer{ts.NewSelect(0)}
	 *   idsTs := dataset.Idx(idsIdx)
	 *   startIdx := []ts.TensorIndexer{ts.NewSelect(3)}
	 *   startTs := dataset.Idx(startIdx)
	 *   endIdx := []ts.TensorIndexer{ts.NewSelect(4)}
	 *   endTs := dataset.Idx(endIdx)
	 *   tk := pretrained.BertLargeCasedWholeWordMaskingSquad()
	 *   padTok := tokenizer.NewAddedToken("[PAD]", true)
	 *   tk.AddSpecialTokens([]tokenizer.AddedToken{padTok})
	 *   featIdx := []ts.TensorIndexer{ts.NewSelect(9)}
	 *   ids := toInt(idsTs.Idx(featIdx).Int64Values())
	 *   start := toInt(startTs.Idx(featIdx).Int64Values())[0]
	 *   end := toInt(endTs.Idx(featIdx).Int64Values())[0]
	 *   context := tk.Decode(ids, true)
	 *   answer := tk.Decode(ids[start:end], true)
	 *   fmt.Printf("context: %q\n", context)
	 *   fmt.Printf("answer: %q\n", answer)
	 *  */
	return dataset
}

func LoadSquadV2Features(datasetName string) []squad.Feature {

	filePath := fmt.Sprintf("/tmp/squadv2-%v.gob", datasetName)

	if isFileNotExist(filePath) {
		fmt.Printf("SQuAD feature data has not been cached yet. Generating and caching...\n")
		loadSquadV2(datasetName, true, false)
	}

	features, err := loadFeatures(filePath)
	if err != nil {
		log.Fatal(err)
	}

	return features
}

// cacheFeatures caches feature data to local file.
func cacheFeatures(features []squad.Feature, filePath string) error {
	outFile, err := os.Create(filePath)
	if err != nil {
		return err
	}
	encoder := gob.NewEncoder(outFile)
	err = encoder.Encode(features)
	if err != nil {
		return err
	}
	outFile.Close()

	return nil
}

// loadFeatures loads SQuAD feature data from local cached file.
func loadFeatures(filePath string) ([]squad.Feature, error) {
	var features []squad.Feature
	inFile, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	decoder := gob.NewDecoder(inFile)
	err = decoder.Decode(&features)
	if err != nil {
		return nil, err
	}
	inFile.Close()

	return features, nil
}

func toInt(input []int64) []int {
	var output []int
	for _, v := range input {
		output = append(output, int(v))
	}

	return output
}

// Check whether train dataset file exist, otherwise, generate it.
// Ref.https://stackoverflow.com/questions/12518876
func isFileNotExist(filePath string) bool {
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return true
	}

	return false
}
