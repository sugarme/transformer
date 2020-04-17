package common

import (
	"log"

	G "gorgonia.org/gorgonia"
)

type Dropout struct {
	dropoutProb float64
}

func NewDropout(p float64) *Dropout {
	return &Dropout{
		dropoutProb: p,
	}
}

func (d *Dropout) ForwardT(input *G.Node, train bool) *G.Node {

	return d.dropout(input, d.dropoutProb, train)
}

func (d *Dropout) dropout(input *G.Node, prob float64, train bool) *G.Node {

	// TODO: implement *trainable* with `train` parameter
	res, err := G.Dropout(input, prob)
	if err != nil {
		log.Fatal(err)
	}

	return res
}
