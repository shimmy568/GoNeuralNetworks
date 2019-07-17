package core

import (
	"gonum.org/v1/gonum/mat"
)

// NeuralNet is a data type that is used to preform basic neural network operations
type NeuralNet struct {
	Weights []mat.Dense
}

// InitNetwork is a function to net up a neural network
func (n NeuralNet) InitNetwork() {
	n.InitNetwork = mat.NewDense()
	// Left off here (im gonna need paper to do this)
}
