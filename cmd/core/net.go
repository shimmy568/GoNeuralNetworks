package core

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// NeuralNet is a data type that is used to preform basic neural network operations
type NeuralNet struct {
	// Importent values about network
	inputCount      int
	outputCount     int
	hiddenLayers    int
	hiddenLayerSize int

	// Weights of the network
	weights []*mat.Dense
}

// generateWeights generates a random set of weights for the creation of the network
func generateWeights(sizeX int, sizeY int) []float64 {
	data := make([]float64, sizeX*sizeY)
	for i := 0; i < sizeX*sizeY; i++ {
		data[i] = rand.NormFloat64()
	}

	return data
}

// sigmoid is an implemention of the sigmoid function for use in .Apply on a matrix of data
func sigmoid(row int, col int, value float64) float64 {
	return math.Pow(math.E, value) / (math.Pow(math.E, value) + 1)
}

// CreateNetwork is a function to create up a neural network
func CreateNetwork(inputCount int, outputCount int, hiddenLayers int, hiddenLayerSize int) NeuralNet {
	n := NeuralNet{}

	n.weights = make([]*mat.Dense, hiddenLayers+1)
	// Copy relevant values to struct
	n.inputCount = inputCount
	n.outputCount = outputCount
	n.hiddenLayers = hiddenLayers
	n.hiddenLayerSize = hiddenLayerSize

	// Generate random weights for network
	for i := 0; i < len(n.weights); i++ {
		if i == 0 {
			n.weights[i] = mat.NewDense(hiddenLayerSize, inputCount, generateWeights(hiddenLayerSize, inputCount))
		} else if i == len(n.weights)-1 {
			n.weights[i] = mat.NewDense(outputCount, hiddenLayerSize, generateWeights(outputCount, hiddenLayerSize))
		} else {
			n.weights[i] = mat.NewDense(hiddenLayerSize, hiddenLayerSize, generateWeights(hiddenLayerSize, hiddenLayerSize))
		}
	}

	return n
}

func printMat(data *mat.Dense) {
	f := mat.Formatted(data, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("mat:\na = % v\n\n", f)
}

func TrainNetwork()

// ForwardPass takes a set of input data and generates a set of output values
func (n NeuralNet) ForwardPass(inputData []float64) []float64 {
	data := mat.NewDense(n.inputCount, 1, inputData)
	md := mat.NewDense(n.hiddenLayerSize, 1, nil)
	o := mat.NewDense(n.outputCount, 1, nil)

	for i := 0; i < n.hiddenLayers+1; i++ {

		if i == n.hiddenLayers {
			// It's the last step in the prop so use o
			o.Product(n.weights[i], md) // Do matrix mult step
			o.Apply(sigmoid, o)         // Do sigmoid step
		} else if i == 0 {
			// First pass so use md to store and data as input
			md.Product(n.weights[i], data) // Do matrix mult step
			md.Apply(sigmoid, md)          // Do sigmoid step
		} else {
			// It's not the last step so use md
			md.Product(n.weights[i], md) // Do matrix mult step
			md.Apply(sigmoid, md)        // Do sigmoid step
		}
	}

	output := make([]float64, n.outputCount)
	for i := 0; i < n.outputCount; i++ {
		output[i] = o.At(i, 0)
	}

	return output
}
