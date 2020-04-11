package ann_test

import (
	"fmt"

	"github.com/pkg/errors"
	. "gorgonia.org/golgi"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// myLayer is a layer with additional support for transformation for shapes.
//
// One may of course do this with a ComposeSeq(Reshape, FC), but this is just for demonstration purposes
type myLayer struct {
	// name is in FC
	FC

	// BE CAREFUL WITH EMBEDDINGS

	// size is in FC and in myLayer
	size int
}

// Model, Name, Type, Shape and Describe are all from the embedded FC

func (l *myLayer) Fwd(a G.Input) G.Result {
	if err := G.CheckOne(a); err != nil {
		return G.Err(errors.Wrapf(err, "Fwd of myLayer %v", l.FC.Name()))
	}
	x := a.Node()
	xShape := x.Shape()

	switch xShape.Dims() {
	case 0, 1:
		return G.Err(errors.Errorf("Unable to handle x of %v", xShape))
	case 2:
		return l.FC.Fwd(x)
	case 3, 4:
		return G.Err(errors.Errorf("NYI"))

	}
	panic("UNIMPLEMENTED")
}

func ConsMyLayer(x G.Input, opts ...ConsOpt) (retVal Layer, err error) {
	l := new(myLayer)
	for _, opt := range opts {
		var o Layer
		var ok bool
		if o, err = opt(l); err != nil {
			return nil, err
		}
		if l, ok = o.(*myLayer); !ok {
			return nil, errors.Errorf("Construction Option returned non *myLayer. Got %T instead", o)
		}
	}
	if err = l.Init(x.(*G.Node)); err != nil {
		return nil, err
	}
	return l, nil
}

func Example_extension() {
	of := tensor.Float64
	g := G.NewGraph()
	x := G.NewTensor(g, of, 4, G.WithName("X"), G.WithShape(100, 1, 28, 28), G.WithInit(G.GlorotU(1)))
	layer, err := ConsMyLayer(x, WithName("EXT"), WithSize(100))
	if err != nil {
		fmt.Printf("Uh oh. Error happened when constructing *myLayer: %v\n", err)
	}
	l := layer.(*myLayer)
	fmt.Printf("Name:  %q\n", l.Name())
	fmt.Printf("Model: %v\n", l.Model())
	fmt.Printf("BE CAREFUL\n======\nl.size is %v. But the models shapes are correct as follows:\n", l.size)
	for _, n := range l.Model() {
		fmt.Printf("\t%v - %v\n", n.Name(), n.Shape())
	}

	// Output:
	// Name:  "EXT"
	// Model: [EXT_W, EXT_B]
	// BE CAREFUL
	// ======
	// l.size is 0. But the models shapes are correct as follows:
	// 	EXT_W - (1, 100)
	// 	EXT_B - (100, 100)
}
