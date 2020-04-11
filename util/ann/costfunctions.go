package ann

import (
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
)

// RMS represents a root mean equare error
// The root-mean-square deviation (RMSD) or root-mean-square error (RMSE) is a frequently
// used measure of the differences between values (sample or population values) predicted
// by a model or an estimator and the values observed.
func RMS(yHat, y G.Input) (retVal *G.Node, err error) {
	if err = G.CheckOne(yHat); err != nil {
		return nil, errors.Wrap(err, "unable to extract node from yHat")
	}

	if err = G.CheckOne(y); err != nil {
		return nil, errors.Wrap(err, "unable to extract node from y")
	}

	if retVal, err = G.Sub(yHat.Node(), y.Node()); err != nil {
		return nil, errors.Wrap(err, "(ŷ-y)")
	}

	if retVal, err = G.Square(retVal); err != nil {
		return nil, errors.Wrap(err, "(ŷ-y)²")
	}

	if retVal, err = G.Mean(retVal); err != nil {
		return nil, errors.Wrap(err, "mean((ŷ-y)²)")
	}

	return
}
