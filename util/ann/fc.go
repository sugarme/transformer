package ann

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	_ ByNamer = &FC{}
)

// WithWB is a FC specific construction option used to initialize a FC.
func WithWB(w, b *G.Node) ConsOpt {
	return func(layer Layer) (Layer, error) {
		fc, ok := layer.(*FC)
		if !ok {
			return layer, errors.Errorf("Expected a *FC. Got %v of %T instead", layer, layer)
		}
		fc.w = w
		fc.b = b
		fc.initialized = true
		return layer, nil
	}
}

// FC represents a fully connected layer
//
// If batched is set to true, then the first dimension is assumed to be the batch dimension
type FC struct {
	w, b *G.Node
	act  ActivationFunction

	name string

	// config
	size        int
	batched     bool
	nobias      bool
	initialized bool
}

// MakeFC creates a FC with the given parameters
func MakeFC(w, b *G.Node, act ActivationFunction, name string, batched bool) FC {
	return FC{
		w:           w,
		b:           b,
		act:         act,
		name:        name,
		batched:     batched,
		initialized: true,
	}
}

// NewFC is the usual way to create a FC
func NewFC(opts ...ConsOpt) *FC {
	retVal := new(FC)
	for _, opt := range opts {
		l, err := opt(retVal)
		if err != nil {
			panic(err)
		}
		retVal = l.(*FC)
	}
	retVal.initialized = true
	return retVal
}

// Model will return the gorgonia.Nodes associated with this fully connected layer
func (l *FC) Model() G.Nodes {
	if l.nobias {
		return G.Nodes{l.w}
	}
	return G.Nodes{l.w, l.b}
}

// Fwd runs the equation forwards
func (l *FC) Fwd(a G.Input) G.Result {
	if err := G.CheckOne(a); err != nil {
		return G.Err(errors.Wrapf(err, "Fwd of FC %v", l.name))
	}

	x := a.Node()
	var xw, xwb *G.Node
	var err error
	if xw, err = G.Mul(x, l.w); err != nil {
		return G.Err(err)
	}

	if l.b == nil {
		xwb = xw
		goto act
	}

	if l.batched && !(l.b.Shape().Eq(xw.Shape())) {
		if xwb, err = G.BroadcastAdd(xw, l.b, nil, []byte{0}); err != nil {
			return G.Err(err)
		}
	} else {
		if xwb, err = G.Add(xw, l.b); err != nil {
			return G.Err(err)
		}
	}
act:
	if l.act == nil {
		return xwb
	}
	return G.LiftResult(l.act(xwb))
}

// Type will return the hm.Type of the fully connected layer
func (l *FC) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b'))
}

// Shape will return the tensor.Shape of the fully connected layer
func (l *FC) Shape() tensor.Shape {
	return l.b.Shape()
}

// Name will return the name of the fully connected layer
func (l *FC) Name() string {
	return l.name
}

// Describe will describe a fully connected layer
func (l *FC) Describe() {
	panic("STUB")
}

// methods to support extensions

// ByName returns a Term by name
func (l *FC) ByName(name string) Term {
	if l.name == name {
		return l
	}
	if l.w.Name() == name {
		return l.w
	}
	if l.b != nil && l.b.Name() == name {
		return l.b
	}
	return nil
}

func (l *FC) Graph() *G.ExprGraph { return l.w.Graph() }

// SetName will set the name of a fully connected layer
func (l *FC) SetName(a string) error { l.name = a; return nil }

// SetSize will set the size of a fully connected layer
func (l *FC) SetSize(a int) error { l.size = a; return nil }

// SetAct will set an activiation function of a fully connected layer
func (l *FC) SetAct(act ActivationFunction) error {
	l.act = act
	return nil
}

// Init will initialize the fully connected layer
func (l *FC) Init(xs ...*G.Node) (err error) {
	x := xs[0]
	g := x.Graph()
	of := x.Dtype()
	X := x
	if x.IsVec() {
		if X, err = G.Reshape(x, tensor.Shape{1, x.Shape()[0]}); err != nil {
			return err
		}
	}
	xshp := X.Shape()
	l.w = G.NewMatrix(g, of, G.WithShape(xshp[1], l.size), G.WithInit(G.GlorotU(1)), G.WithName(l.name+"_W"))
	switch {
	case l.batched && !l.nobias:
		l.b = G.NewMatrix(g, of, G.WithShape(1, l.size), G.WithInit(G.Zeroes()), G.WithName(l.name+"_B"))
	case !l.batched && !l.nobias:
		l.b = G.NewMatrix(g, of, G.WithShape(xshp[0], l.size), G.WithInit(G.Zeroes()), G.WithName(l.name+"_B"))
	}
	l.initialized = true
	return nil
}

// ConsFC is a FC construction function. It takes a gorgonia.Input that has a *gorgonia.Node.
func ConsFC(in G.Input, opts ...ConsOpt) (retVal Layer, err error) {
	x := in.Node()
	if x == nil {
		return nil, errors.Errorf("ConsFC expects a *Node. Got input %v of  %T instead", in, in)
	}

	inshape := x.Shape()
	if inshape.Dims() > 2 || inshape.Dims() == 0 {
		return nil, errors.Errorf("Expected shape is either a vector or a matrix")
	}

	// construct
	l := &FC{}
	for _, opt := range opts {
		var o Layer
		var ok bool
		if o, err = opt(l); err != nil {
			return nil, err
		}
		if l, ok = o.(*FC); !ok {
			return nil, errors.Errorf("Construction Option returned a non FC. Got %T instead", o)
		}
	}

	// prep
	if err = l.Init(x); err != nil {
		return nil, err
	}

	return l, nil
}
