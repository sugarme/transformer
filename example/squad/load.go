package main

import (
	"fmt"
	"log"
	"reflect"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func loadCheckPoint(filePath string, vs *nn.VarStore, device gotch.Device) {
	namedTensors, err := ts.LoadMultiWithDevice(filePath, device)
	if err != nil {
		log.Fatal(err)
	}

	vars := make(map[string]*ts.Tensor, 0)
	for _, x := range namedTensors {
		vars[x.Name] = x.Tensor
	}

	for k, v := range vs.Vars.NamedVariables {
		if _, ok := vars[k]; !ok {
			fmt.Printf("Not found (from saved file) Ts name: %v. Skipping...\n", k)
			continue
		}
		a := v.MustSize()
		b := vars[k].MustSize()
		if !reflect.DeepEqual(a, b) {
			fmt.Printf("Mismatched shape: Ts name: %q - VarStore: %v - Saved File: %v\n", k, a, b)
			continue
		} else {
			ts.NoGrad(func() {
				vs.Vars.NamedVariables[k].Copy_(vars[k])
			})
		}
	}
}
