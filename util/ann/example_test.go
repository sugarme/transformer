package ann_test

import (
	"fmt"

	. "gorgonia.org/golgi"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func softmax(a *gorgonia.Node) (*gorgonia.Node, error) { return gorgonia.SoftMax(a) }

func Example() {
	n := 100
	of := tensor.Float64
	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, of, 4, gorgonia.WithName("X"), gorgonia.WithShape(n, 1, 28, 28), gorgonia.WithInit(gorgonia.GlorotU(1)))
	y := gorgonia.NewMatrix(g, of, gorgonia.WithName("Y"), gorgonia.WithShape(n, 10), gorgonia.WithInit(gorgonia.GlorotU(1)))
	nn, err := ComposeSeq(
		x,
		L(ConsReshape, ToShape(n, 784)),
		L(ConsFC, WithSize(50), WithName("l0"), AsBatched(true), WithActivation(gorgonia.Tanh), WithBias(true)),
		L(ConsDropout, WithProbability(0.5)),
		L(ConsFC, WithSize(150), WithName("l1"), AsBatched(true), WithActivation(gorgonia.Rectify)), // by default WithBias is true
		L(ConsLayerNorm, WithSize(20), WithName("Norm"), WithEps(0.001)),
		L(ConsFC, WithSize(10), WithName("l2"), AsBatched(true), WithActivation(softmax), WithBias(false)),
	)
	if err != nil {
		panic(err)
	}
	out := nn.Fwd(x)
	if err = gorgonia.CheckOne(out); err != nil {
		panic(err)
	}

	cost := gorgonia.Must(RMS(out, y))
	model := nn.Model()
	if _, err = gorgonia.Grad(cost, model...); err != nil {
		panic(err)
	}
	m := gorgonia.NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		panic(err)
	}

	fmt.Printf("Model: %v\n", model)
	// Output:
	// Model: [l0_W, l0_B, l1_W, l1_B, Norm_W, Norm_B, l2_W]
}
