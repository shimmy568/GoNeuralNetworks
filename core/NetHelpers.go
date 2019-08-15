package core

import (
	"errors"
	"math/rand"

	"github.com/shimmy568/GoNeuralNetworks/data"

	"gonum.org/v1/gonum/mat"
)

// This file stores some of the helper functions for the NeuralNetwork struct

// The function sets up the arrays for the forward propagation process
func (n NeuralNet) initMatrixes(inpData []float64) (inputData *mat.Dense, middleData *mat.Dense, outputData *mat.Dense, err error) {
	// Check that input data is right length
	if len(inpData) != n.inputCount {
		return nil, nil, nil, errors.New("input data is incorrect length for network")
	}

	inputData = mat.NewDense(n.inputCount, 1, inpData)

	middleData = mat.NewDense(n.hiddenLayerSize, 1, nil)
	outputData = mat.NewDense(n.outputCount, 1, nil)

	return inputData, middleData, outputData, nil
}

// executeLayer is a function that executes a layer of the neural network
// The function will modify the arguments to do it's work
// ------------
// 	layerNum:		The layer that the function will execute
// 	inputData:	The input data to the network. Should have dims of (n.inputCount, 1)
// 	middleData:	A matrix that will be used to store the data when executing the middle layers.
// 							Should have dims of (n.hiddenLayerSize, 1)
// 	outputData:	A matrix that will be used to store the output data from the network
// 							Should have dims of (n.outputCount, 1)
// ------------
// The function will decide which *mat.Dense to use depending on what layer you are executing
// To run an entire network operation you will need to give all inputs of layerNum from
// 0 - n.hiddenLayers
func (n NeuralNet) executeLayer(
	layerNum int,
	inputData *mat.Dense,
	middleDataInput *mat.Dense,
	middleDataOutput *mat.Dense,
	outputData *mat.Dense,
) {
	// Create temp dense to store data in

	if layerNum == n.hiddenLayers {
		// It's the last step in the prop so use o
		outputData.Product(n.weights[layerNum], middleDataInput) // Do matrix mult step
		outputData.Apply(sigmoidWrapper, outputData)             // Do sigmoid step
	} else if layerNum == 0 {
		// First pass so use md to store and data as input
		middleDataOutput.Product(n.weights[layerNum], inputData) // Do matrix mult step
		middleDataOutput.Apply(sigmoidWrapper, middleDataOutput) // Do sigmoid step
	} else {
		// It's not the last step so use md
		middleDataOutput.Product(n.weights[layerNum], middleDataInput) // Do matrix mult step
		middleDataOutput.Apply(sigmoidWrapper, middleDataOutput)       // Do sigmoid step
	}
}

// executeLayerReverse does the same thing as executeLayer but uses an inverse sigmoid function instead of sigmoid
// This allows us to run the error through the network backwards easily
func (n NeuralNet) executeLayerReverse(
	layerNum int,
	inputData *mat.Dense,
	middleDataInput *mat.Dense,
	middleDataOutput *mat.Dense,
	outputData *mat.Dense,
) {
	if layerNum == n.hiddenLayers {
		// It's the last step in the prop so use o
		outputData.Product(n.weights[layerNum], middleDataInput) // Do matrix mult step
		outputData.Apply(sigmoidInverseWrapper, outputData)      // Do sigmoid step
	} else if layerNum == 0 {
		// First pass so use md to store and data as input
		middleDataOutput.Product(n.weights[layerNum], inputData)        // Do matrix mult step
		middleDataOutput.Apply(sigmoidInverseWrapper, middleDataOutput) // Do sigmoid step
	} else {
		// It's not the last step so use md
		middleDataOutput.Product(n.weights[layerNum], middleDataInput)  // Do matrix mult step
		middleDataOutput.Apply(sigmoidInverseWrapper, middleDataOutput) // Do sigmoid step
	}
}

// generateWeights generates a random set of weights for the creation of the network
func generateWeights(sizeX int, sizeY int) []float64 {
	data := make([]float64, sizeX*sizeY)
	for i := 0; i < sizeX*sizeY; i++ {
		data[i] = rand.NormFloat64()
	}

	return data
}

// sigmoidPrimeWrapper is a wrapper for use in mat.Apply
func sigmoidPrimeWrapper(row int, col int, value float64) float64 {
	return sigmoidPrime(value)
}

// sigmoidWrapper is a function that wraps the sigmoid function to allow for use is mat.Apply
func sigmoidWrapper(row int, col int, value float64) float64 {
	return sigmoid(value)
}

// sigmoidInverseWrapper is a wrapper for the sigmoidInverse for the use in mat.Apply
func sigmoidInverseWrapper(row int, col int, value float64) float64 {
	return sigmoidInverse(value)
}

// backPropIter is a function that does the process for a single iteration of backprop
// ------------
// 	layerIndex:	The index of the layer that we are adjusting the weights for
// 	inputInfo:	The input for the layer that we are adjusting for
// 	outputInfo:	The outputs for the layer that we are adjusting for
// 	layerErr:		The error for the layer
// ------------
// This function will change the weights of the neural network but shouldn't have any extra side effects
func (n NeuralNet) backPropIter(layerIndex int, inputInfo *mat.Dense, outputInfo *mat.Dense, layerErr *mat.Dense) {
	tmp := mat.NewDense(n.hiddenLayerSize, n.hiddenLayerSize, nil) // Create tmp mat for calculations
	outputCopy := mat.DenseCopyOf(outputInfo)                      // Copy mdo into tmp matrix
	inputCopy := mat.DenseCopyOf(inputInfo)                        // Copy mdi into tmp matrix
	outputCopy.Apply(sigmoidPrimeWrapper, outputCopy)              // Apply sigmoid prime to mdo
	tmp.Mul(layerErr, outputCopy)                                  // Multiply layer error and mdoCopy
	inputCopy.Product(tmp, inputCopy)                              // Take dot product of layer input values and tmp
	inputCopy.Scale(n.learningRate, inputCopy)                     // Scale the error adjustment by the learning rate
	n.weights[layerIndex].Add(inputCopy, n.weights[layerIndex])    // Adjust the weights
}

// vectorizeMatrix takes a 2D matrix with many columns and turns it into a matrix with only 1 row (vector)
func vectorizeMatrix(matrix *mat.Dense) *mat.VecDense {
	// Init vector
	width, height := matrix.Dims()
	output := mat.NewVecDense(width*height, nil)

	// Copy data from matrix to vector
	for col := 0; col < width; col++ {
		for row := 0; row < height; row++ {
			output.SetVec(col+row, matrix.At(col, row))
		}
	}

	// Return vector
	return output
}

// TrainMonochromeImage trains a neural network
func (n *NeuralNet) TrainMonnochromeImage(image *data.MonochromeImageData, expectedOutput *mat.VecDense) (err error) {
	// Check that the image is the right size
	imageMat := image.GetDense()
	imageWidth, imageHeight := imageMat.Dims()
	if imageWidth*imageHeight != n.inputCount {
		return errors.New("input image is incorrect size for number of input nodes in the network")
	}

	// Check that expectedOutput is right size
	if expectedOutput.Len() != n.outputCount {
		return errors.New("expected output is incorrect size for number of output nodes in the network")
	}

	// Turn matrix into vector and turn into training item struct
	item := CreateTrainingItem(vectorizeMatrix(imageMat), expectedOutput)

	// Train network with vector
	n.backProp(item)

	return nil
}

// PredictMonochromeImage predicts an output using a network given an image
func (n *NeuralNet) PredictMonochromeImage(image *data.MonochromeImageData) (output *mat.VecDense, err error) {
	// Check that the image is the right size
	imageMat := image.GetDense()
	imageWidth, imageHeight := imageMat.Dims()
	if imageWidth*imageHeight != n.inputCount {
		return nil, errors.New("input image is incorrect size for number of input nodes in the network")
	}

	vecData := vectorizeMatrix(image.GetDense())

	// Copy vector to float array
	arrayData := make([]float64, vecData.Len())
	for i := 0; i < vecData.Len(); i++ {
		arrayData[i] = vecData.AtVec(i)
	}

	return n.Predict(arrayData)
}
