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
func (n *NeuralNet) Predict(inputData []float64) (*mat.VecDense, error) {
	// Create matrices and check for error
	data, md, o, err := n.initMatrixes(inputData)
	if err != nil {
		return nil, err
	}

	for i := 0; i < n.hiddenLayers+1; i++ {
		// Just use md as input and output since it doesn't matter for this useage
		// Since we don't need to keep track of input and output layer data
		n.executeLayer(i, data, md, md, o)
	}

	// Copy output data to an array
	output := mat.NewVecDense(n.outputCount, nil)
	for i := 0; i < n.outputCount; i++ {
		output.SetVec(i, o.At(i, 0))
	}

	return output, nil
}

// The inverse of the sigmoid function
func sigmoidInverse(value float64) float64 {
	return math.Log(value / (1 - value))
}

// sigmoidPrime function is a function that represents the derivative of the sigmoid function
func sigmoidPrime(value float64) float64 {
	return sigmoid(1 - sigmoid(value))
}

// backProp is a function that is for one iteration of training using backpropagation
func (n *NeuralNet) backProp(item *TrainingItem) error {
	// Check training item matches network
	if n.inputCount != len(item.inputData) {
		return fmt.Errorf("Input dimension for training data doesn't match network's")
	}
	if n.outputCount != len(item.expectedOutput) {
		return fmt.Errorf("Output dimension for training data doesn't match network's")
	}

	// Run forward pass for network
	res, err := n.Predict(item.inputData)
	if err != nil {
		return err
	}

	// Find error from expected value
	layerError := mat.NewDense(n.outputCount, 1, nil)
	for i := 0; i < n.outputCount; i++ {
		layerError.Set(i, 0, res.AtVec(i)-item.expectedOutput[i])
	}

	// Run the errors in reverse and place the final values in firstLayerError for use in backprop
	_, firstLayerError, _, err := n.initMatrixes(item.inputData)
	if err != nil {
		return err
	}
	n.executeLayerReverse(n.hiddenLayers-1, nil, nil, firstLayerError, layerError) // Execute the last layer in reverse

	for i := n.hiddenLayers - 2; i > 0; i-- { // Don't execute the last and first layers of the network
		n.executeLayerReverse(i, nil, firstLayerError, firstLayerError, nil) // Execute the layer of the network in reverse
	}

	// mdi is middle layer input mdo is middle layer output since we need to keep track of both
	// for the backprop process
	data, mdi, o, err := n.initMatrixes(item.inputData)
	if err != nil {
		return err
	}
	mdo := mat.DenseCopyOf(mdi)

	// creating temp mats that are used in backprop math
	for i := 0; i < n.hiddenLayers; i++ {
		n.executeLayer(i, data, mdi, mdo, o)
		terr := mat.DenseCopyOf(firstLayerError)

		// Move error forward a level before we adjust the weights for the layer
		n.executeLayer(i+1, nil, firstLayerError, firstLayerError, nil)

		// Adjust weights for layer
		if i == n.hiddenLayers {
			// If network is on last layer use mdi/o for input/output
			n.backPropIter(i, mdi, o, terr)

			// Don't bother copying the data from mdi to mdo since it's the last step anyhow
			break
		} else if i == 0 {
			// If on first layer use data/mdo for input/output
			n.backPropIter(i, data, mdo, terr)
		} else {
			// If network is not on last/first layer use mdi/mdo for input/output
			n.backPropIter(i, mdi, mdo, terr)
		}

		// Copy data from mdo to mdi
		mdi.Copy(mdo)
	}

	return nil
}

// TrainMultiple is a function that trains the network given a set of training data
func (n *NeuralNet) TrainMultiple(trainingData []*TrainingItem) error {
	for i := 0; i < len(trainingData); i++ {
		err := n.backProp(trainingData[i])

		if err != nil {
			return err
		}
	}

	return nil
}
