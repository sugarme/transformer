package bert

import (
	ts "github.com/sugarme/gotch/tensor"
)

// TensorOpt is a function type to create pointer to tensor.
type TensorOpt func() *ts.Tensor

func MaskTensorOpt(t *ts.Tensor) TensorOpt {
	return func() *ts.Tensor {
		return t
	}
}

func EncoderMaskTensorOpt(t *ts.Tensor) TensorOpt {
	return func() *ts.Tensor {
		return t
	}
}

func EncoderHiddenStateTensorOpt(t *ts.Tensor) TensorOpt {
	return func() *ts.Tensor {
		return t
	}
}
