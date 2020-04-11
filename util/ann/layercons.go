package ann

import (
	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
)

// LayerCons makes a layer
type LayerCons func(input G.Input, opts ...ConsOpt) (Layer, error)

type consThunk struct {
	LayerCons
	Opts []ConsOpt
}

// L is a thunk of creation function
func L(cons LayerCons, opts ...ConsOpt) Term { return consThunk{cons, opts} }

func (t consThunk) Name() string { return "thunk" }
func (t consThunk) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b'))
}
