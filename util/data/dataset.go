package data

import ()

// Dataset defines a set of methods to operate
// on a map (from keys to data samples)
// 1. Initiate a dataset
// 2. Iterate elements from dataset
type Dataset interface {
	GetItem() // fetch a data sample for a given key
	Len()     // return size of the dataset
}

type MapDataset struct {
	data map[interface{}]interface{}
}

// IterableDataset represents an interatble data samples
type IterableDataset struct {
	data map[interface{}]interface{}
}

// Implement iterator interface
func (itd *IterableDataset) Next() (interface{}, bool) {

	// TODO: implement this
	return "", true
}

func (itd *IterableDataset) Add() {}

// Implement Dataset
func (itd *IterableDataset) GetItem() {}
func (itd *IterableDataset) Len()     {}

// // TODO:
// type TensorDataset struct{}
// type ConcatDataset struct{}
// type ChainDataset struct{}
// type Subset struct{}
