package nn_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/sermo/util/nn"
)

func TestNn_New(t *testing.T) {

	var device nn.Device

	vs := nn.NewVarStore(device)
	root := vs.Root()
	e1 := root.Entry("key")
	t1 := e1.OrZeros([]int{3, 1, 4})

	e2 := root.Entry("key")
	t2 := e2.OrZeros([]int{1, 5, 9, 10})

	// let t1 = root.entry("key").or_zeros(&[3, 1, 4]);
	// let t2 = root.entry("key").or_zeros(&[1, 5, 9]);

	if !reflect.DeepEqual(t1.Size(), t2.Size()) {
		t.Errorf("T1 Size: %v\n", t1.Size())
		t.Errorf("T2 Size: %v\n", t2.Size())
	}

}

func TestNn_Save(t *testing.T) {
}
