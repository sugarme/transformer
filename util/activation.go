package util

import (
	ts "github.com/sugarme/gotch/tensor"
)

// ActivationFn is an activation function.
type ActivationFn interface {
	// Fwd is a forward pass through x.
	Fwd(x *ts.Tensor) *ts.Tensor
	Name() string
}

// ReLU activation:
// ===============

type ReluActivation struct {
	name string
}

var Relu = ReluActivation{}

func NewRelu() *ReluActivation {
	return &ReluActivation{"relu"}
}

func (r *ReluActivation) Fwd(x *ts.Tensor) *ts.Tensor {
	return x.MustRelu(false)
}

func (r *ReluActivation) Name() string {
	return r.name
}

// GeLU activation:
// ===============

type GeluActivation struct {
	name string
}

var Gelu = GeluActivation{}

func NewGelu() *GeluActivation {
	return &GeluActivation{"gelu"}
}

func (g *GeluActivation) Fwd(x *ts.Tensor) *ts.Tensor {
	return x.MustGelu(false)
}

func (g *GeluActivation) Name() string {
	return g.name
}

// Tanh activation:
// ===============

type TanhActivation struct {
	name string
}

var Tanh = TanhActivation{}

func NewTanh() *TanhActivation {
	return &TanhActivation{"tanh"}
}

func (t *TanhActivation) Fwd(x *ts.Tensor) *ts.Tensor {
	return x.MustTanh(false)
}

func (t *TanhActivation) Name() string {
	return t.name
}

// Swish activation:
// ===============

type SwishActivation struct {
	name string
}

var Swish = SwishActivation{}

func NewSwish() *SwishActivation {
	return &SwishActivation{"swish"}
}

func (s *SwishActivation) Fwd(x *ts.Tensor) *ts.Tensor {
	return x.Swish()
}

func (s *SwishActivation) Name() string {
	return s.name
}

// Mish activation:
// =================

type MishActivation struct {
	name string
}

var Mish = MishActivation{}

func NewMish() *MishActivation {
	return &MishActivation{"mish"}
}

func (m *MishActivation) Fwd(x *ts.Tensor) *ts.Tensor {
	softplus := x.MustSoftplus(false)
	tanh := softplus.MustTanh(true)
	retVal := x.MustMm(tanh, false)
	tanh.MustDrop()
	return retVal
}

func (m *MishActivation) Name() string {
	return m.name
}

func geluNew(xs *ts.Tensor) *ts.Tensor {
	// TODO: implement
	// x * 0.5 * (((x.pow(3.0f64) * 0.044715 + x) * ((2f64 / PI).sqrt())).tanh() + 1)
	// return retVal
	panic("not implemeted yet")
}

var ActivationFnMap map[string]ActivationFn = map[string]ActivationFn{
	"gelu":  NewGelu(),
	"relu":  NewRelu(),
	"tanh":  NewTanh(),
	"swish": NewSwish(),
	"mish":  NewMish(),
}
