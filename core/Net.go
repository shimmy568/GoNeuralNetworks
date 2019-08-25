package core

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// NeuralNet is a data type that is used to preform basic neural network operations
type NeuralNet struct {
	// Importent values about network
	inputCount      int
	outputCount     int
	hiddenLayers    int
	hiddenLayerSize int
	learningRate    float64

	// Weights of the network
	weights []*mat.Dense
}

// sigmoid is an implemention of the sigmoid function for use in .Apply on a matrix of data
func sigmoid(value float64) float64 {
	return math.Pow(math.E, value) / (math.Pow(math.E, value) + 1)
}

// CreateNetwork is a function to create up a neural network
func CreateNetwork(inputCount int, outputCount int, hiddenLayers int, hiddenLayerSize int, learningRate float64) NeuralNet {
	n := NeuralNet{}

	n.weights = make([]*mat.Dense, hiddenLayers+1)
	// Copy relevant values to struct
	n.inputCount = inputCount
	n.outputCount = outputCount
	n.hiddenLayers = hiddenLayers
	n.hiddenLayerSize = hiddenLayerSize
	n.learningRate = learningRate

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

// Predict takes a set of input data and generates a set of output values
func (n *NeuralNet) Predict(inputData []float64) *mat.VecDense {

	inputs := mat.NewDense(len(inputData), 1, inputData)
	var hiddenInputs mat.Matrix
	var hiddenOutputs mat.Matrix

	for i := 0; i < n.hiddenLayers; i++ {
		if i == 0 {
			hiddenInputs = dot(n.weights[i], inputs)
		} else {
			hiddenInputs = dot(n.weights[i], hiddenOutputs)
		}

		hiddenOutputs = apply(sigmoidWrapper, hiddenInputs)
	}

	// Copy data to VecDense object
	output := mat.NewVecDense(n.outputCount, nil)
	for i := 0; i < n.outputCount; i++ {
		output.SetVec(i, hiddenOutputs.At(i, 0))
	}

	return output
}

// sigmoidPrime applies a inverse of the sigmoid function to a matrix
func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m)) // m * (1 - m)
}

// Train is a function that is for one iteration of training using backpropagation
func (n *NeuralNet) Train(item *TrainingItem) error {
	// Check training item matches network
	if n.inputCount != len(item.inputData) {
		return fmt.Errorf("Input dimension for training data doesn't match network's")
	}
	if n.outputCount != len(item.expectedOutput) {
		return fmt.Errorf("Output dimension for training data doesn't match network's")
	}

	hiddenInputData := make([]mat.Matrix, n.hiddenLayers+1)
	hiddenOutputData := make([]mat.Matrix, n.hiddenLayers+1)

	input := mat.NewDense(n.inputCount, 1, item.inputData)

	// Do the forward propagation step and store all the results from each layer for the backpropagation step
	for i := 0; i < n.hiddenLayers+1; i++ {
		if i == 0 {
			hiddenInputData[i] = dot(n.weights[i], input)
		} else {
			hiddenInputData[i] = dot(n.weights[i], hiddenOutputData[i-1])
		}

		hiddenOutputData[i] = apply(sigmoidWrapper, hiddenInputData[i])
	}

	// Init arrays and find the error for the output layer of the network
	targets := mat.NewDense(len(item.expectedOutput), 1, item.expectedOutput)
	errors := make([]mat.Matrix, n.hiddenLayers+1)
	errors[n.hiddenLayers] = subtract(targets, hiddenOutputData[len(hiddenOutputData)-1])

	// Find the errors for the rest of the layers
	for i := 0; i < n.hiddenLayers; i++ {
		// Take the dot product of the weights and the error from the previous layer
		errors[n.hiddenLayers-(1+i)] = dot(n.weights[n.hiddenLayers-i].T(), errors[n.hiddenLayers-i])
	}

	// Do the actual backpropagation for the weights
	for i := 0; i < n.hiddenLayers+1; i++ {

		// Find what the input for the layer is
		var layerInput mat.Matrix
		if i == 0 { // If it's the first layer use the inputs for the network
			layerInput = input
		} else { // If it's any other layer use the output from the previous layer
			layerInput = hiddenOutputData[i-1]
		}

		delta := scale(n.learningRate,
			dot(multiply(errors[i], sigmoidPrime(hiddenOutputData[i])),
				layerInput.T()))
		n.weights[i] = add(n.weights[i], delta).(*mat.Dense)
	}

	return nil
}

// TrainMultiple is a function that trains the network given a set of training data
func (n *NeuralNet) TrainMultiple(trainingData []*TrainingItem) error {
	for i := 0; i < len(trainingData); i++ {
		err := n.Train(trainingData[i])

		if err != nil {
			return err
		}
	}

	return nil
}
