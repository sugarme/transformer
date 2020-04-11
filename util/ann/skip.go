package ann

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type skip struct {
	b *G.Node
}

func ConsSkip(_ G.Input, opts ...ConsOpt) (retVal Layer, err error) {
	l := &skip{}
	for _, opt := range opts {
		var o Layer
		var ok bool
		if o, err = opt(l); err != nil {
			return nil, err
		}
		if l, ok = o.(*skip); !ok {
			return nil, errors.Errorf("Construction option does not return *skip. Got %v of %T instead", o, o)
		}
	}
	return l, nil
}

func (l *skip) Model() G.Nodes { return nil }

func (l *skip) Fwd(x G.Input) G.Result {
	if err := G.CheckOne(x); err != nil {
		return G.Err(err)
	}
	return G.TransformResult(x, l.b)(G.Add(x.Node(), l.b))
}

func (l *skip) Name() string { return "+" + l.b.Name() }

func (l *skip) Type() hm.Type { return hm.NewFnType(l.b.Type(), l.b.Type()) }

func (l *skip) Shape() tensor.Shape { return l.b.Shape() }

func (l *skip) Describe() {}
