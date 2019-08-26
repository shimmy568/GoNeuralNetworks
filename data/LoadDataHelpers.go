package data

import (
	"bufio"
	"errors"
	"os"
	"path/filepath"

	"gonum.org/v1/gonum/mat"

	"github.com/shimmy568/GoNeuralNetworks/util"
)

// This file holds any helper functions that are used in the process of loading data

// ListFilesDir is a function that gets a list of strings from a file
// This function is used in the loading of the image paths for the data sets
func ListFilesDir(filePath string) ([]string, error) {
	absPath, err := filepath.Abs(filePath)

	file, err := os.Open(absPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	return lines, scanner.Err()
}

// SegmentDataSet breaks a list of strings into two sepperate data sets
// pathsA will contain len(paths) * ratio elements it it
// ratio must be 0 <= ratio <= 1 or will return error
func SegmentDataSet(paths []string, ratio float64) (pathsA []string, pathsB []string, err error) {
	indices := make(map[int]bool) // The map that will hold the indices for pathsA
	r := util.GetRand()

	// Check that ratio value is valid
	if ratio < 0 || ratio > 1 {
		return nil, nil, errors.New("Ratio argument was outside of valid range")
	}

	// Calculate the number of elements we need in pathsA and pathsB and init arrays
	lenPathsA := int(float64(len(paths)) * ratio)
	lenPathsB := len(paths) - lenPathsA

	// Create the paths arrays and a map to keep track
	pathsA = make([]string, lenPathsA)
	pathsB = make([]string, lenPathsB)

	// Randomly select lenPathsA indices and keep track of them in a set so we can add them to A
	for i := 0; i < lenPathsA; i++ {
		indexToPick := -1

		// Keep picking random indices until you find a new one
		for true {
			indexToPick = int(r.Float64() * float64(len(paths)))
			if _, ok := indices[indexToPick]; !ok {
				break
			}
		}

		// Add the new index to the map
		indices[indexToPick] = true
	}

	// Put the rest of the elements in pathsB
	indexA := 0
	indexB := 0
	for i := 0; i < len(paths); i++ {
		// Check if i is in map
		if _, ok := indices[i]; ok {
			// i is in map add to A
			pathsA[indexA] = paths[i]
			indexA++
		} else {
			// i is not in map add to B
			pathsB[indexB] = paths[i]
			indexB++
		}
	}

	return pathsA, pathsB, nil
}

// PrefixStringArray takes an array of strings and applies a prefix string to all contained strings
func PrefixStringArray(strs []string, prefix string) []string {
	// Create output array
	outputData := make([]string, len(strs))

	for i := 0; i < len(strs); i++ {
		outputData[i] = prefix + strs[i]
	}

	return outputData
}

// MakeDenseNormal makes all the values in the matrix between 0 and 1 (exclusivily)
func MakeDenseNormal(matrix *mat.Dense) {
	min := mat.Min(matrix)
	max := mat.Max(matrix)

	// Subtract min from all values
	matrix.Apply(func(i, j int, value float64) float64 {
		return value - min
	}, matrix)

	// Divide all values by max and make sure the values aren't equal to exsactly 0 or 1
	matrix.Apply(func(i, j int, value float64) float64 {
		return 0.999*(value/max) + 0.001
	}, matrix)
}
