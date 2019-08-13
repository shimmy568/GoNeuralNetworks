package core

import (
	"gonum.org/v1/gonum/mat"
)

// TrainingItem is a struct that represents on set of training data
type TrainingItem struct {
	inputData      []float64
	expectedOutput []float64
}

// CreateTrainingItem creates a training item struct given the data as input
func CreateTrainingItem(inputData *mat.VecDense, expectedData *mat.VecDense) (output *TrainingItem) {
	// Create new arrays
	tmpInput := make([]float64, inputData.Len())
	tmpExpected := make([]float64, expectedData.Len())

	// Copy input data to tmpInput
	for i := 0; i < inputData.Len(); i++ {
		tmpInput[i] = inputData.AtVec(i)
	}

	// Copy expected data to tmpExpected
	for i := 0; i < expectedData.Len(); i++ {
		tmpExpected[i] = expectedData.AtVec(i)
	}

	// Create the training item struct
	output = &TrainingItem{
		inputData:      tmpInput,
		expectedOutput: tmpExpected,
	}

	return output
}
