package nn

import (
	"math/rand"

	// G "gorgonia.org/gorgonia"
	ts "gorgonia.org/tensor"
)

type initEnum int

const (
	// Constant value
	Float64 = iota

	// Random normal
	RandNorm

	// Uniform initialization between some lower and upper bounds
	Uniform

	// Kaiming uniform initialization
	KaimingUniform
)

var maxInit = KaimingUniform

type UniformT struct {
	Lo float64
	Up float64
}

func NewUniform(lo, up float64) UniformT {
	return UniformT{
		Lo: lo,
		Up: up,
	}
}

// Init represents an initialization function
type InitT interface{}

var internalMaps = map[initEnum]InitT{
	Float64:        rand.Float64,
	RandNorm:       rand.NormFloat64,
	Uniform:        NewUniform,
	KaimingUniform: NewUniform,
}

// Init initializes a new float tensor with a specified shape, device
// and initialization value
func Init(i InitT, dims []int, device Device) ts.Tensor {

	// dt := ts.Of(i.(ts.Dtype))
	dt := ts.Of(ts.Float64)
	shape := ts.WithShape(len(dims))
	// TODO: init with device

	return ts.New(dt, shape)

}

/* /// Creates a new float tensor with the specified shape, device, and initialization.
 * pub fn init(i: Init, dims: &[i64], device: Device) -> Tensor {
 *     match i {
 *         Init::Const(cst) => {
 *             // Optimize the case for which a single C++ code can be done.
 *             if cst == 0. {
 *                 Tensor::zeros(dims, (Kind::Float, device))
 *             } else if (cst - 1.).abs() <= std::f64::EPSILON {
 *                 Tensor::ones(dims, (Kind::Float, device))
 *             } else {
 *                 Tensor::ones(dims, (Kind::Float, device)) * cst
 *             }
 *         }
 *         Init::Uniform { lo, up } => Tensor::zeros(dims, (Kind::Float, device)).uniform_(lo, up),
 *         Init::Randn { mean, stdev } => {
 *             if mean == 0. && (stdev - 1.).abs() <= std::f64::EPSILON {
 *                 Tensor::randn(dims, (Kind::Float, device))
 *             } else {
 *                 Tensor::randn(dims, (Kind::Float, device)) * stdev + mean
 *             }
 *         }
 *         Init::KaimingUniform => {
 *             let fan_in: i64 = dims.iter().skip(1).product();
 *             let bound = (1.0 / fan_in as f64).sqrt();
 *             Tensor::zeros(dims, (Kind::Float, device)).uniform_(-bound, bound)
 *         }
 *     }
 * } */
