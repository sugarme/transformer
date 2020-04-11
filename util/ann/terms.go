package ann

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

var (
	_ Term = (Layer)(nil)
	_ Term = Name("")
	_ Term = (*Env)(nil)
)

// Term represents a term that can be used in Golgi
type Term interface {
	Name() string
}

// Name is a variable by name
type Name string

// Name will return itself as a string
func (n Name) Name() string { return string(n) }

// Env is a linked list representing an environment.
// Within the documentation, an environment is written as such:
// 	e := (x ↦ X)
// `x` is the name while `X` is the *gorgonia.Node
//
// A longer environment may look like this:
//	e2 := (x ↦ X :: y ↦ Y)
//	                ^
// Here, e2 is pointing to the *Env that contains (y ↦ Y).
//
// When talking about Envs in general, it will often be written as a meta variable σ.
type Env struct {
	name string
	node *G.Node
	prev *Env
}

// NewEnv creates a new Env.
func NewEnv(name string, node *G.Node) *Env {
	return &Env{name: name, node: node}
}

func (e *Env) hinted(prealloc G.Nodes) {
	prealloc = append(prealloc, e.node)
	if e.prev != nil {
		e.prev.hinted(prealloc)
	}
}

// Extend allows users to extend the environment.
//
// Given an environment as follows:
// 	e := (x ↦ X)
// if `e.Extend(y, Y)` is called, the following is returned
//	e2 := (x ↦ X :: y ↦ Y)
//	                ^
// The pointer will be pointing to the *Env starting at y
func (e *Env) Extend(name string, node *G.Node) *Env {
	return &Env{name: name, node: node, prev: e}
}

// ByName returns the first node that matches the given name. It also returns the parent
//
// For example, if we have an Env as follows:
// 	e := (x ↦ X1 :: w ↦ W :: x ↦ X2)
// 	                         ^
//
// The caret indicates the pointer of *Env. Now, if e.ByName("x") is called,
// then the result returned will be X2 and (x ↦ X1 :: w ↦ W)
func (e *Env) ByName(name string) (*G.Node, *Env) {
	if e.name == name {
		return e.node, e.prev
	}
	if e.prev != nil {
		return e.prev.ByName(name)
	}
	return nil, nil
}

// Model will return the gorgonia.Nodes associated with this environment
func (e *Env) Model() G.Nodes {
	retVal := G.Nodes{e.node}
	if e.prev != nil {
		retVal = append(retVal, e.prev.Model()...)
	}
	return retVal
}

// HintedModel will return the gorgonia.Nodes hinted associated with this environment
func (e *Env) HintedModel(hint int) G.Nodes {
	prealloc := make(G.Nodes, 0, hint)
	e.hinted(prealloc)
	return prealloc
}

// Name will return the name of the composition
func (e *Env) Name() string {
	var name string
	if e.prev != nil {
		name = e.prev.Name() + " :: "
	}
	name += fmt.Sprintf("%v ↦ %v", e.name, e.node)
	return name
}

type tag struct{ a, b Term }
