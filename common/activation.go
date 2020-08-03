package common

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// ActivationFn is an activation function.
type ActivationFn interface {
	// Fwd is a forward pass through x.
	Fwd(x *G.Node) (*G.Node, error)

	// Clone the activation.
	Clone() ActivationFn

	Name() string
}

// SigmoidActivation is a sigmoid activation layer.
type SigmoidActivation struct {
	name string
}

// Sigmoid activation function.
var Sigmoid = &SigmoidActivation{}

// NewSigmoid returns a new sigmoid activation layer.
func NewSigmoid() *SigmoidActivation {
	return &SigmoidActivation{"sigmoid"}
}

// Fwd is a forward pass through the layer.
func (s *SigmoidActivation) Fwd(x *G.Node) (*G.Node, error) {
	return G.Sigmoid(x)
}

func (s *SigmoidActivation) Name() string {
	return s.name
}

// Learnables returns all learnable nodes within this layer.
func (s *SigmoidActivation) Learnables() (n G.Nodes) {
	return n
}

// Compile the layer.
func (s *SigmoidActivation) Compile(x *G.Node, opts ...CompileOpt) {}

// Clone the activation.
func (s *SigmoidActivation) Clone() ActivationFn {
	return NewSigmoid()
}

// TanhActivation is a tanh activation layer.
type TanhActivation struct {
	name string
}

// Tanh activation.
var Tanh = &TanhActivation{}

// NewTanh returns a new tanh activation layer.
func NewTanh() *TanhActivation {
	return &TanhActivation{"tanh"}
}

// Fwd is a forward pass through the layer.
func (t *TanhActivation) Fwd(x *G.Node) (*G.Node, error) {
	return G.Tanh(x)
}

func (t *TanhActivation) Name() string {
	return t.name
}

// Learnables returns all learnable nodes within this layer.
func (t *TanhActivation) Learnables() (n G.Nodes) {
	return n
}

// Compile the layer.
func (t *TanhActivation) Compile(x *G.Node, opts ...CompileOpt) {}

// Clone the activation.
func (t *TanhActivation) Clone() ActivationFn {
	return NewTanh()
}

// ReLUActivation is a relu activation layer.
type ReLUActivation struct {
	name string
}

// ReLU activation.
var ReLU = &ReLUActivation{}

// NewReLU returns a new relu activation layer.
func NewReLU() *ReLUActivation {
	return &ReLUActivation{"relu"}
}

// Fwd is a forward pass through the layer.
func (r *ReLUActivation) Fwd(x *G.Node) (*G.Node, error) {
	return G.Rectify(x)
}

func (r *ReLUActivation) Name() string {
	return r.name
}

// Learnables returns all learnable nodes within this layer.
func (r *ReLUActivation) Learnables() (n G.Nodes) {
	return n
}

// Compile the layer.
func (r *ReLUActivation) Compile(x *G.Node, opts ...CompileOpt) {}

// Clone the activation.
func (r *ReLUActivation) Clone() ActivationFn {
	return NewReLU()
}

// LeakyReLUActivation is a leaky relu activation layer.
type LeakyReLUActivation struct {
	alpha float64
	name  string
}

// LeakyReLU is default leaky relu activation.
var LeakyReLU = &LeakyReLUActivation{0.01, "leakyRelu"}

// NewLeakyReLU returns a new leaky relu activation layer.
func NewLeakyReLU(alpha float64) *LeakyReLUActivation {
	return &LeakyReLUActivation{alpha: alpha}
}

// Fwd is a forward pass through the layer.
func (r *LeakyReLUActivation) Fwd(x *G.Node) (*G.Node, error) {
	return G.LeakyRelu(x, r.alpha)
}

func (r *LeakyReLUActivation) Name() string {
	return r.name
}

// Learnables returns all learnable nodes within this layer.
func (r *LeakyReLUActivation) Learnables() (n G.Nodes) {
	return n
}

// Compile the layer.
func (r *LeakyReLUActivation) Compile(x *G.Node, opts ...CompileOpt) {}

// Clone the activation.
func (r *LeakyReLUActivation) Clone() ActivationFn {
	return NewLeakyReLU(r.alpha)
}

// SoftmaxActivation is a softmax activation layer.
type SoftmaxActivation struct {
	axis []int
	name string
}

// Softmax is the default softmax activation.
var Softmax = &SoftmaxActivation{}

// NewSoftmax returns a new leaky softmax activation layer.
func NewSoftmax(axis ...int) *SoftmaxActivation {
	// if len(axis) == 0 {
	// 	axis = append(axis, 0)
	// }
	return &SoftmaxActivation{axis: axis, name: "softmax"}
}

// Fwd is a forward pass through the layer.
func (s *SoftmaxActivation) Fwd(x *G.Node) (*G.Node, error) {
	// fmt.Printf("running softmax with x shape: %v dims: %v \n", x.Shape(), x.Dims())
	return softMax(x, s.axis...)
}

func (s *SoftmaxActivation) Name() string {
	return s.name
}

// Learnables returns all learnable nodes within this layer.
func (s *SoftmaxActivation) Learnables() (n G.Nodes) {
	return n
}

// Compile the layer.
func (s *SoftmaxActivation) Compile(x *G.Node, opts ...CompileOpt) {}

// Clone the activation.
func (s *SoftmaxActivation) Clone() ActivationFn {
	return NewSoftmax(s.axis...)
}

// LinearActivation is a linear (identity) activation layer.
type LinearActivation struct {
	name string
}

// Linear activation.
var Linear = &LinearActivation{}

// NewLinear is a linear activation layer.
func NewLinear() *LinearActivation {
	return &LinearActivation{"linear"}
}

// Fwd is a forward pass through the layer.
func (l *LinearActivation) Fwd(x *G.Node) (*G.Node, error) {
	return x, nil
}

func (l *LinearActivation) Name() string {
	return l.name
}

// Learnables returns all learnable nodes within this layer.
func (l *LinearActivation) Learnables() (n G.Nodes) {
	return n
}

// Compile the layer.
func (l *LinearActivation) Compile(x *G.Node, opts ...CompileOpt) {}

// Clone the activation.
func (l *LinearActivation) Clone() ActivationFn {
	return NewLinear()
}

// softMax performs softmax on the input. Specifically this is used:
//		e^(a[i]) / sum((e^(a[i])))
// For a more numerically stable SoftMax, use StableSoftMax.
//
// This is ripped from Gorgonia core and tweaked as there was a bug in it https://github.com/gorgonia/gorgonia/issues/373
// which is currently being worked on.
func softMax(a *G.Node, axes ...int) (retVal *G.Node, err error) {
	aShape := a.Shape()

	if aShape[0] == 1 {
		aShape = aShape[1:]
		a, err = G.Reshape(a, aShape)
		log.Printf("a reshaped to %v", a.Shape())
	}
	axis := aShape.Dims() - 1 // default: last dim
	if a.IsColVec() || (a.IsVector() && !a.IsRowVec()) {
		axis = 0
	}

	if len(axes) > 0 {
		if axes[0] >= axis+1 || axes[0] < 0 {
			return nil, fmt.Errorf("Cannot perform SoftMax on axis %d. Input has shape %v", axes[0], a.Shape())
		}
		axis = axes[0]
	}
	var exp, sum *G.Node
	if exp, err = G.Exp(a); err != nil {
		return nil, err
	}
	if sum, err = G.Sum(exp, axis); err != nil {
		return nil, err
	}

	if sum.IsScalar() {
		return G.HadamardDiv(exp, sum)
	}

	// reshape if necessary
	ss := sum.Shape()
	diff := exp.Shape().Dims() - ss.Dims()

	// TODO: multirank softmax
	if diff > 0 {
		newShape := ts.Shape(ts.BorrowInts(ss.Dims() + diff))
		copy(newShape, ss)
		copy(newShape[axis+1:], newShape[axis:])
		newShape[axis] = 1

		if sum, err = G.Reshape(sum, newShape); err != nil {
			return nil, fmt.Errorf("Failed to reshape: %v", err)
		}
	}
	retVal, err = G.BroadcastHadamardDiv(exp, sum, nil, []byte{byte(axis)})
	if err != nil {
		return
	}
	return

}
