package nn

import (
	"math"
	"math/rand"
	"reflect"
	"time"

	rng "github.com/leesper/go_rng"
	// G "gorgonia.org/gorgonia"
	ts "gorgonia.org/tensor"
)

// InitWFn returns a function type which helps to
// initiate weight values for tensor
type InitT interface{}

func InitFloat64(f float64) float64 {
	return f
}

// InitUniform returns a []float64 drawn from a uniform distribution between [low, high) that is provided
func NewUniform(low, high float64, s ...int) []float64 {
	size := ts.Shape(s).TotalSize()

	rand := rng.NewUniformGenerator(time.Now().UnixNano())
	retVal := make([]float64, size)
	for i := range retVal {
		retVal[i] = rand.Float64Range(low, high)
	}
	return retVal
}

func NewKaimingUniform(dims []int) []float64 {

	fanIn := factorial(uint64(len(dims) - 1))
	bound := math.Sqrt(1.0 / float64(fanIn))

	return NewUniform(-bound, bound)

}

func factorial(n uint64) (result uint64) {
	if n > 0 {
		result = n * factorial(n-1)
		return result
	}
	return 1
}

func NewRandnStandard() float64 {
	return rand.NormFloat64()
}

func NewRandn(mean, std float64) float64 {
	return rand.NormFloat64()*std + mean
}

// Init initializes a new float tensor with a specified shape, device
// and initialization value
func Init(i InitT, dims []int, device Device) ts.Tensor {

	// TODO: init with device
	// dt := ts.Of(i.(ts.Dtype))
	dt := ts.Of(ts.Float64)
	shape := ts.WithShape(dims...)

	vals := make([]float64, ts.ProdInts(dims))

	switch reflect.TypeOf(i).String() {
	case "float64":
		// TODO: improve performance for big size
		for idx := range vals {
			vals[idx] = reflect.ValueOf(i).Float()
		}

	case "[]float64":
		vals = i.([]float64)
	}

	return ts.New(dt, shape, ts.WithBacking(vals))

}
