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
	// Create the training item struct
	output = &TrainingItem{
		inputData:      inputData,
		expectedOutput: expectedData,
	}

	return output
}
