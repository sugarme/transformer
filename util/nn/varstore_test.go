package nn_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/sermo/util"
	"github.com/sugarme/sermo/util/nn"

	ts "gorgonia.org/tensor"
)

func TestNn_New(t *testing.T) {

	var device nn.Device

	vs := nn.NewVarStore(device)
	root := vs.Root()
	e1 := root.Entry("key")
	t1 := e1.OrZeros([]int{3, 2, 4})

	e2 := root.Entry("key")
	t2 := e2.OrZeros([]int{1, 5, 9})

	wantT1Shape := ts.Shape{3, 2, 4}

	if !reflect.DeepEqual(t1.Shape(), wantT1Shape) {
		t.Errorf("Want: %v\n", wantT1Shape)
		t.Errorf("Got: %v\n", t1.Shape())
	}
	if !reflect.DeepEqual(t2.Shape(), wantT1Shape) {
		t.Errorf("Want: %v\n", wantT1Shape)
		t.Errorf("Got: %v\n", t2.Shape())
	}

}

// Test tensor initiation
func TestNn_Init(t *testing.T) {
	vs := nn.NewVarStore(nn.CPU)
	root := vs.Root()
	tensor := root.Zeros("t1", []int{3})

	want := []float64{0.0, 0.0, 0.0}
	got := tensor.Data()

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want t1: %v\n", want)
		t.Errorf("Got t1: %v\n", got)
	}

	tensor = root.Var("t2", []int{3}, float64(0.0))
	got = tensor.Data()

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want t2: %v\n", want)
		t.Errorf("Got t2: %v\n", got)
	}

	tensor = root.Var("t3", []int{3}, float64(1.0))
	got = tensor.Data()
	want = []float64{1.0, 1.0, 1.0}

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want t3: %v\n", want)
		t.Errorf("Got t3: %v\n", got)
	}

	tensor = root.Var("t4", []int{3}, float64(0.5))
	got = tensor.Data()
	want = []float64{0.5, 0.5, 0.5}

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want t4: %v\n", want)
		t.Errorf("Got t4: %v\n", got)
	}

	tensor = root.Var("t4", []int{2}, float64(42.0))
	got = tensor.Data()
	want = []float64{42.0, 42.0}

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want t4: %v\n", want)
		t.Errorf("Got t4: %v\n", got)
	}

	uniform := root.Var("t5", []int{100}, nn.InitUniform(1.0, 2.0, 100))
	data := uniform.Data()

	min, max := util.MinMaxFloat64(data.([]float64))

	if min < 1.0 || min > 2.0 {
		t.Errorf("Want: > 1.0\n")
		t.Errorf("Got: %v\n", min)
	}

	if max < 1.0 || max > 2.0 {
		t.Errorf("Want: < 2.0\n")
		t.Errorf("Got: %v\n", max)
	}

	tensor = root.Randn("normal", []int{2}, 42.0, 0.15)
	data = tensor.Data()
	val := data.([]float64)[0]

	if val > 41.85 {
		t.Errorf("Want: < 41.85\n")
		t.Errorf("Got: %v\n", val)
	}

	tensor = root.RandnStandard("normal", []int{2})
	data = tensor.Data()
	val = data.([]float64)[0]

	if val < -0.15 || val > 0.15 {
		t.Errorf("Want: > -0.15 or < 0.15\n")
		t.Errorf("Got: %v\n", val)
	}
}
