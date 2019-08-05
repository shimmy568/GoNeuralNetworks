package core

import "testing"

// TestInitMatrixes is the unit test for initMatrixes
func TestInitMatrixes(t *testing.T) {
	// Set up test data
	n := CreateNetwork(5, 6, 1, 7, 0.1)
	inputData := []float64{1, 2, 3, 4, 5}

	// Execute test
	inp, middle, output, err := n.initMatrixes(inputData)
	if err != nil {
		t.Error("Function threw error")
	}

	// Check output
	inpHeight, _ := inp.Dims()
	if inpHeight != len(inputData) {
		t.Error("Input layer array generated is wrong width")
	}

	middleWidth, _ := middle.Dims()
	if middleWidth != 7 {
		t.Error("Middle layer array generated is wrong width")
	}

	outputWidth, _ := output.Dims()
	if outputWidth != 6 {
		t.Error("Output layer array generated is wrong width")
	}
}
