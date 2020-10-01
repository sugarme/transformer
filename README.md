# Transformer [![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)[![Go.Dev reference](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white&style=flat-square)](https://pkg.go.dev/github.com/sugarme/transformer?tab=doc)[![Travis CI](https://api.travis-ci.org/sugarme/transformer.svg?branch=master)](https://travis-ci.org/sugarme/transformer)[![Go Report Card](https://goreportcard.com/badge/github.com/sugarme/transformer)](https://goreportcard.com/report/github.com/sugarme/transformer) 

## Overview

`transformer` is pure Go package to facilitate applying Natural Language Processing (NLP) models train/test and inference in Go. 

This package is in active mode of building and there are many change ahead. Hence you can use it with your complete own risk. The package will be considered as stable when version 1.0 is released. 

`transformer` is heavily inspired by and based on the popular [Python HuggingFace Transformers](https://github.com/huggingface/transformers). It's also influenced by [Rust version - rust-bert](https://github.com/guillaume-be/rust-bert). In fact, all pre-trained models for Rust are compatible to import to this Go `transformer` package as both `rust-bert`'s dependency Pytorch Rust binding - [**`tch-rs`**](https://github.com/LaurentMazare/tch-rs) and Go binding [**`gotch`**](https://github.com/sugarme/gotch) are built with similar principles. 

`transformer` is part of an ambitious goal (together with [**tokenizer**](https://github.com/sugarme/tokenizer) and [**gotch**](https://github.com/sugarme/gotch)) to bring more AI/deep-learning tools to Gophers so that they can stick to the language they love and good at and build faster software in production. 

## Dependencies

2 main dependency are:

- `tokenizer`
- `gotch`

## Prerequisites and installation

- As this package depends on `gotch` which is a Pytorch C++ API binding for Go, a pre-compiled Libtorch copy (CPU or GPU) should be installed in your machine. Please follow [gotch](https://github.com/sugarme/gotch) installation instruction for detail. 
- Install package: `go get -u github.com/sugarme/transformer`

## Basic example

```go
    import (
        "fmt"
        "log"

        "github.com/sugarme/gotch"
        ts "github.com/sugarme/gotch/tensor"
        "github.com/sugarme/tokenizer"

        "github.com/sugarme/transformer/bert"
    )

    func main() {
        var config *bert.BertConfig = new(bert.BertConfig)
        if err := transformer.LoadConfig(config, "bert-base-uncased", nil); err != nil {
            log.Fatal(err)
        }

        var model *bert.BertForMaskedLM = new(bert.BertForMaskedLM)
        if err := transformer.LoadModel(model, "bert-base-uncased", config, nil, gotch.CPU); err != nil {
            log.Fatal(err)
        }

        var tk *bert.Tokenizer = bert.NewTokenizer()
        if err := tk.Load("bert-base-uncased", nil); err != nil{
            log.Fatal(err)
        }

        sentence1 := "Looks like one [MASK] is missing"
        sentence2 := "It was a very nice and [MASK] day"

        var input []tokenizer.EncodeInput
        input = append(input, tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(sentence1)))
        input = append(input, tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(sentence2)))

        encodings, err := tk.EncodeBatch(input, true)
        if err != nil {
            log.Fatal(err)
        }

        var maxLen int = 0
        for _, en := range encodings {
            if len(en.Ids) > maxLen {
                maxLen = len(en.Ids)
            }
        }

        var tensors []ts.Tensor
        for _, en := range encodings {
            var tokInput []int64 = make([]int64, maxLen)
            for i := 0; i < len(en.Ids); i++ {
                tokInput[i] = int64(en.Ids[i])
            }

            tensors = append(tensors, ts.TensorFrom(tokInput))
        }

        inputTensor := ts.MustStack(tensors, 0).MustTo(device, true)
        var output ts.Tensor
        ts.NoGrad(func() {
            output, _, _ = model.ForwardT(inputTensor, ts.None, ts.None, ts.None, ts.None, ts.None, ts.None, false)
        })
        index1 := output.MustGet(0).MustGet(4).MustArgmax(0, false, false).Int64Values()[0]
        index2 := output.MustGet(1).MustGet(7).MustArgmax(0, false, false).Int64Values()[0]

        got1, ok := tk.IdToToken(int(index1))
        if !ok {
            fmt.Printf("Cannot find a corresponding word for the given id (%v) in vocab.\n", index1)
        }
        got2, ok := tk.IdToToken(int(index2))
        if !ok {
            fmt.Printf("Cannot find a corresponding word for the given id (%v) in vocab.\n", index2)
        }

        fmt.Println(got1)
        fmt.Println(got2)
        
        // Output:
        // person
        // pleasant
    }
```

## Getting Started

- See [pkg.go.dev](https://pkg.go.dev/github.com/sugarme/transformer?tab=doc) for detail APIs 


## License

`transformer` is Apache 2.0 licensed.


## Acknowledgement

- This project has been inspired and used many concepts from [Python HuggingFace Transformers](https://github.com/huggingface/transformers) and [Rust version - rust-bert](https://github.com/guillaume-be/rust-bert). 

- Pre-trained models and configurations are downloaded remotely from HuggingFace. 





