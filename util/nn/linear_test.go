package nn_test

import (
	// "reflect"
	"testing"
	// "github.com/sugarme/sermo/util/nn"
	// ts "gorgonia.org/tensor"
)

func TestNn_NewLinear(t *testing.T) {
	// TODO

	// var device nn.Device
	//
	// vs := nn.NewVarStore(device)
	// root := vs.Root()
	// e1 := root.Entry("key")
	// t1 := e1.OrZeros([]int{3, 2, 4})
	//
	// e2 := root.Entry("key")
	// t2 := e2.OrZeros([]int{1, 5, 9})
	//
	// wantT1Shape := ts.Shape{3, 2, 4}
	//
	// if !reflect.DeepEqual(t1.Shape(), wantT1Shape) {
	// t.Errorf("Want: %v\n", wantT1Shape)
	// t.Errorf("Got: %v\n", t1.Shape())
	// }
	// if !reflect.DeepEqual(t2.Shape(), wantT1Shape) {
	// t.Errorf("Want: %v\n", wantT1Shape)
	// t.Errorf("Got: %v\n", t2.Shape())
	// }

}

/* fn optimizer_test() {
 *     tch::manual_seed(42);
 *     // Create some linear data.
 *     let xs = Tensor::of_slice(&(1..15).collect::<Vec<_>>())
 *         .to_kind(Kind::Float)
 *         .view([-1, 1]);
 *     let ys = &xs * 0.42 + 1.337;
 *
 *     // Fit a linear model (with deterministic initialization) on the data.
 *     let vs = nn::VarStore::new(Device::Cpu);
 *     let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
 *     let cfg = nn::LinearConfig {
 *         ws_init: nn::Init::Const(0.),
 *         bs_init: Some(nn::Init::Const(0.)),
 *         bias: true,
 *     };
 *     let mut linear = nn::linear(vs.root(), 1, 1, cfg);
 *
 *     let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
 *     let initial_loss = f64::from(&loss);
 *     assert!(initial_loss > 1.0, "initial loss {}", initial_loss);
 *
 *     // Optimization loop.
 *     for _idx in 1..50 {
 *         let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
 *         opt.backward_step(&loss);
 *     }
 *     let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
 *     let final_loss = f64::from(loss);
 *     assert!(final_loss < 0.25, "final loss {}", final_loss);
 *
 *     // Reset the weights to their initial values.
 *     tch::no_grad(|| {
 *         linear.ws.init(nn::Init::Const(0.));
 *         linear.bs.init(nn::Init::Const(0.));
 *     });
 *     let initial_loss2 = f64::from(xs.apply(&linear).mse_loss(&ys, Reduction::Mean));
 *     assert_eq!(initial_loss, initial_loss2)
 * } */
