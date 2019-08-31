package core

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// NeuralNet is a data type that is used to preform basic neural network operations
type NeuralNet struct {
	// Importent values about network
	inputCount      int
	outputCount     int
	hiddenLayers    int
	hiddenLayerSize int

	// LearningRate is the learning rate used when training the network
	LearningRate float64

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
	n.LearningRate = learningRate

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

// GetInputCount returns the number of input nodes for the network
func (n *NeuralNet) GetInputCount() int {
	return n.inputCount
}

// GetOutputCount returns the number of input nodes for the network
func (n *NeuralNet) GetOutputCount() int {
	return n.outputCount
}

// GetHiddenLayerCount returns the number of input nodes for the network
func (n *NeuralNet) GetHiddenLayerCount() int {
	return n.hiddenLayers
}

// GetHiddenLayerSize the size of the hidden layers in the network
func (n *NeuralNet) GetHiddenLayerSize() int {
	return n.hiddenLayerSize
}

// SaveWeights save the weights of the network to a file on disk
func (n *NeuralNet) SaveWeights(path string) error {
	dataFile, _ := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0662)
	writer := csv.NewWriter(bufio.NewWriter(dataFile))
	defer dataFile.Close()

	// Write network metadata in first row of csv
	inputCount := strconv.Itoa(n.inputCount)
	outputCount := strconv.Itoa(n.outputCount)
	hiddenLayerCount := strconv.Itoa(n.hiddenLayers)
	hiddenLayerSize := strconv.Itoa(n.hiddenLayerSize)
	info := []string{inputCount, outputCount, hiddenLayerCount, hiddenLayerSize}
	writer.Write(info)

	// Convert all layers of network to string array
	data := make([]string, 0)
	for i := range n.weights {
		// Convert layer to btye array
		rawData, err := n.weights[i].MarshalBinary()
		if err != nil {
			return err
		}

		// Add byte array to data array as string
		data = append(data, string(rawData))
	}

	// Write layer data to network
	writer.Write(data)

	return nil
}

// LoadWeights load the weights of the network from a file on disk
func (n *NeuralNet) LoadWeights(path string) error {
	dataFile, _ := os.Open(path)
	reader := csv.NewReader(bufio.NewReader(dataFile))
	defer dataFile.Close()

	rawMetadata, err := reader.Read()
	if err != nil {
		return err
	}

	// The array that holds the parsed metadata
	// Spots in the array are as follows [inputCount, outputCount, hiddenLayerCount, hiddenLayerSize]
	metadata := make([]int, 4)
	for i := range rawMetadata {
		metadata[i], err = strconv.Atoi(rawMetadata[i])
		if err != nil {
			return err
		}
	}

	// make sure the loaded data matches the dims of the network we are loading it into
	if metadata[0] != n.inputCount {
		return errors.New("Input count of loaded model doesn't match the network")
	}

	if metadata[1] != n.outputCount {
		return errors.New("Output count of loaded model doesn't match the network")
	}

	if metadata[2] != n.hiddenLayers {
		return errors.New("Hidden layer count doesn't match the network")
	}

	if metadata[3] != n.hiddenLayerSize {
		return errors.New("Hidden layer size doesn't match the network")
	}

	// Read the layer data from the file
	layerData, err := reader.Read()
	if err != nil {
		return err
	}

	// Parse the layer data from the file
	for i := range layerData {
		n.weights[i].UnmarshalBinary([]byte(layerData[i]))
	}

	return nil
}

// Predict takes a set of input data and generates a set of output values
func (n *NeuralNet) Predict(inputData []float64) *mat.VecDense {

	inputs := mat.NewDense(len(inputData), 1, inputData)
	var hiddenInputs mat.Matrix
	var hiddenOutputs mat.Matrix

	for i := 0; i < n.hiddenLayers+1; i++ {
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

	var hiddenInputData mat.Matrix
	hiddenOutputData := make([]mat.Matrix, n.hiddenLayers+1)

	input := mat.NewDense(n.inputCount, 1, item.inputData)

	// Do the forward propagation step and store all the results from each layer for the backpropagation step
	for i := 0; i < n.hiddenLayers+1; i++ {
		if i == 0 {
			hiddenInputData = dot(n.weights[i], input)
		} else {
			hiddenInputData = dot(n.weights[i], hiddenOutputData[i-1])
		}

		hiddenOutputData[i] = apply(sigmoidWrapper, hiddenInputData)
	}

	// Init arrays and find the error for the output layer of the network
	targets := mat.NewDense(len(item.expectedOutput), 1, item.expectedOutput)
	errors := make([]mat.Matrix, n.hiddenLayers+1)
	errors[n.hiddenLayers] = subtract(targets, hiddenOutputData[len(hiddenOutputData)-1]) // Find error for output layer

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

		delta := dot(multiply(errors[i], sigmoidPrime(hiddenOutputData[i])),
			layerInput.T())
		delta.Scale(n.LearningRate, delta)
		n.weights[i].Add(n.weights[i], delta)
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
