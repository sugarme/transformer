package ann

import "gorgonia.org/gorgonia"

// ActivationFunction represents an activation function
// Note: This may become an interface once we've worked through all the linter errors
type ActivationFunction func(*gorgonia.Node) (*gorgonia.Node, error)
