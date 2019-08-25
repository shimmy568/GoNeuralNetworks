package core

import (
	"fmt"
	"math"

	"github.com/shimmy568/GoNeuralNetworks/util"

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

// sigmoidPrime function is a function that represents the derivative of the sigmoid function
func sigmoidPrime(value float64) float64 {
	return sigmoid(1 - sigmoid(value))
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

	// Run forward pass for network
	res := n.Predict(item.inputData)

	// Find error from expected value
	layerError := mat.NewDense(n.outputCount, 1, nil)
	for i := 0; i < n.outputCount; i++ {
		layerError.Set(i, 0, res.AtVec(i)-item.expectedOutput[i])
	}

	util.PrintFloatArray(item.expectedOutput)
	fmt.Println("res: ")
	util.PrintMatrix(res)

	fmt.Println("layerError: ")
	util.PrintMatrix(layerError)

	fmt.Println("---------------")

	// Run the errors in reverse and place the final values in firstLayerError for use in backprop
	_, firstLayerError, _, err := n.initMatrixes(item.inputData)
	if err != nil {
		return err
	}
	n.executeLayerReverse(n.hiddenLayers, nil, nil, firstLayerError, layerError) // Execute the last layer in reverse

	for i := n.hiddenLayers - 1; i > 0; i-- { // Don't execute the last and first layers of the network
		n.executeLayerReverse(i, nil, firstLayerError, firstLayerError, nil) // Execute the layer of the network in reverse
	}

	// mdi is middle layer input mdo is middle layer output since we need to keep track of both
	// for the backprop process
	data, mdi, o, err := n.initMatrixes(item.inputData)
	if err != nil {
		return err
	}
	mdo := mat.DenseCopyOf(mdi)

	// Loop for the backpropagation logic
	for i := 0; i < n.hiddenLayers; i++ {

		if i != 0 {
			n.executeLayer(i+1, nil, firstLayerError, firstLayerError, nil) // wtf is this line LMAO
		}

		n.executeLayer(i, data, mdi, mdo, o)
		terr := mat.DenseCopyOf(firstLayerError)

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
		err := n.Train(trainingData[i])

		if err != nil {
			return err
		}
	}

	return nil
}
