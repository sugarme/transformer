package nn

import (
	G "gorgonia.org/gorgonia"
)

// Optimizer defines a struct to run gradient descent.
type Optimizer struct {
	Opt                  G.Solver
	Variables            Variables
	VariablesInOptimizer uint
	Config               interface{}
}

// OptimizerConfig is used to build optimizer
type OptimizerConfig interface {
	buildCOpt(lr float64) *G.Solver
	build(vs VarStore, lr float64) *Optimizer
}

/* // Build builds an optimizer with the specified learning rate handling variables
 * // stored in `varstore`
 * func (oc *OptimizerConfig) Build(vs VarStore, lr float64) *Optimizer {
 *   opt := oc.buildCOpt(lr)
 *
 *   vs.Variables.Mut.Lock()
 *   defer vs.Variables.Mut.Unlock()
 *   v := vs.Variables
 *
 *   opt.AddParameters(v.TrainableVariables) // method of COptimizer
 *
 *   return &Optimizer{
 *     Opt:                  opt,
 *     Variables:            v,
 *     VariablesInOptimizer: uint(len(v.TrainableVariables)),
 *     Config:               oc,
 *   }
 * } */

// SGD optimizer:
// ==============

// SGD is parameters for SGD optimizer
// NOTE: SGD will implement `OptimizerConfig` interface
type SGD struct { // Should it be named as `SGDConfig`?
	Momentum  float64 // momentum factor
	Dampening float64 // dampening for momentum
	Wd        float64 // weigth decay - (L2 penalty)
	Nesterov  bool    // enables Nesterov momentum
}

func DefaultSGD() *SGD {
	return &SGD{
		Momentum:  0.0,
		Dampening: 0.0,
		Wd:        0.0,
		Nesterov:  false,
	}
}

func NewSGD(momentum, dampening, wd float64, nesterov bool) *SGD {
	return &SGD{momentum, dampening, wd, nesterov}
}

// Implement OptimizerConfig for SGD
func (s *SGD) buildCOpt(lr float64) G.Solver {
	withLR := G.WithLearnRate(lr)
	withMomentum := G.WithMomentum(s.Momentum)
	withL2 := G.WithL2Reg(s.Wd)
	// TODO: implement `switch` to initiate either
	// - `VanillarSolver`
	// - `Momentum`

	sol := G.NewVanillaSolver(withLR, withMomentum, withL2)
	return sol
}
