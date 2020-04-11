package common

import (
// ts "gorgonia.org/tensor"
)

type Dropout struct {
	dropoutProb float64
}

func NewDropout(p float64) *Dropout {
	return &Dropout{
		dropoutProb: p,
	}
}

// func (d *Dropout) ForwardT(input ts.Tensor, train bool) ts.Tensor {
// return input.Dropout(d.dropoutProb, train)
// }

/* impl ModuleT for Dropout {
 *     fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
 *         input.dropout(self.dropout_prob, train)
 *     }
 * } */
