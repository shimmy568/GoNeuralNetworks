package core

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/stat/distuv"

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

// generateWeights generates a random set of weights for the creation of the network
func generateWeights(sizeX int, sizeY int) []float64 {

	d := distuv.Uniform{
		Min: -1 / math.Sqrt(float64(sizeY)),
		Max: 1 / math.Sqrt(float64(sizeY)),
	}

	data := make([]float64, sizeX*sizeY)
	for i := 0; i < sizeX*sizeY; i++ {
		data[i] = d.Rand()
	}

	return data
}

// sigmoidWrapper is a function that wraps the sigmoid function to allow for use is mat.Apply
func sigmoidWrapper(row int, col int, value float64) float64 {
	return sigmoid(value)
}

// vectorizeMatrix takes a 2D matrix with many columns and turns it into a matrix with only 1 row (vector)
func vectorizeMatrix(matrix *mat.Dense) *mat.VecDense {
	// Init vector
	width, height := matrix.Dims()
	output := mat.NewVecDense(width*height, nil)

	// Copy data from matrix to vector
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			output.SetVec((row*width)+col, matrix.At(col, row))
		}
	}

	// Return vector
	return output
}

// TrainMonnochromeImage trains a neural network
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
	n.Train(item)

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

	return n.Predict(arrayData), nil
}
