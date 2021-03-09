# BERT NER model


## Pretrained model

This is an example of using pretrained BERT NER model for inference.

- The pretrained model is from [kamalkraj/BERT-NER](https://github.com/kamalkraj/BERT-NER)
- To convert Pytorch model .bin to Gotch compatible format, use the following steps:

**Python: Convert model weights to npz**

```python
from pathlib import Path
import shutil
import os
import numpy as np
import torch
import subprocess

target_path = Path().absolute()
os.makedirs(str(target_path), exist_ok=True)
weights = torch.load("./pytorch_model.bin", map_location='cpu')
nps = {}
for k, v in weights.items():
    k = k.replace("gamma", "weight").replace("beta", "bias")
    nps[k] = np.ascontiguousarray(v.cpu().numpy())

np.savez(target_path / 'model.npz', **nps)
```

**Go Gotch: load model from numpy**

```go
package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

var task string

func init() {
	flag.StringVar(&task, "task", "", "specify a task, ie., 'convert', 'check'")
}

func main() {
	flag.Parse()
	switch task {
	case "convert":
		convert()
	case "check":
		checkModel("./bert-ner.gt")
	default:
		panic("Unspecified or invalid task. ")
	}
}

// convert converts numpy model weights to `gotch` model weights.
func convert() {
	filepath := "./model.npz"

	namedTensors, err := ts.ReadNpz(filepath)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Num of named tensor: %v\n", len(namedTensors))
	outputFile := "bert-ner.gt"
	err = ts.SaveMultiNew(namedTensors, outputFile)
	if err != nil {
		log.Fatal(err)
	}
}

// checkModel loads model weights from file and prints out tensor names.
func checkModel(file string) {
	vs := nn.NewVarStore(gotch.CPU)
	err := vs.Load(file)

	namedTensors, err := ts.LoadMultiWithDevice(file, vs.Device())
	if err != nil {
		log.Fatal(err)
	}

	// var namedTensorsMap map[string]*ts.Tensor = make(map[string]*ts.Tensor, 0)
	for _, namedTensor := range namedTensors {
		// namedTensorsMap[namedTensor.Name] = namedTensor.Tensor
		fmt.Println(namedTensor.Name)
		fmt.Printf("%s - size: %v\n", namedTensor.Name, namedTensor.Tensor.MustSize())
	}
}
```

## Run inference

```go
go run .
```

**Input**
"Steve went to Paris"

**Output**

```bash

Steve      (B-PER, p=0.99950)
went       (O    , p=0.99983)
to         (O    , p=0.99983)
Paris      (B-LOC, p=0.99893)

```

