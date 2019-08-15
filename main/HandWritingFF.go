package main

import (
	"errors"
	"fmt"
	"log"
	"path/filepath"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"

	"github.com/shimmy568/GoNeuralNetworks/core"
	"github.com/shimmy568/GoNeuralNetworks/data"
)

// This file is for holding the logic associated with having

func runHandwritingFF() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	// TODO Implement pseudo code

	// Get list of images to load from disk
	paths, err := data.ListFilesDir("/home/owen/Documents/datasets/handwriting/English/Hnd/all.txt~")
	if err != nil {
		panic(err)
	}

	// Apply the prefix to paths loaded from text file
	paths = data.PrefixStringArray(paths, "/home/owen/Documents/datasets/handwriting/English/Hnd/")

	// Segment data set into a training set and a testing set
	trainingSet, testingSet, _ := data.SegmentDataSet(paths, 0.1)

	// Filter and label the data
	trainingSetLabels, trainingSet, err := filterAndLabelData(trainingSet)
	if err != nil {
		log.Fatal(err)
	}

	testingSetLabels, testingSet, err := filterAndLabelData(testingSet)
	if err != nil {
		log.Fatal(err)
	}

	// Generate the expected vectors for each label for testing and training
	trainingSetExpectedData := generateExpectedOutputFromLables(trainingSetLabels)
	// testingSetExpectedData := generateExpectedOutputFromLables(testingSetLabels)

	// Load images from disk
	trainingImages, err := data.LoadMonochromeImages(trainingSet, 88, 66)
	if err != nil {
		log.Fatal(err)
	}

	testingImages, err := data.LoadMonochromeImages(testingSet, 88, 66)
	if err != nil {
		log.Fatal(err)
	}

	w, h := trainingImages[0].GetDense().Dims()
	fmt.Printf("Width of training images: %d,%d\n", w, h)

	w, h = testingImages[0].GetDense().Dims()
	fmt.Printf("Width of testing images: %d,%d\n", w, h)

	fmt.Printf("Number of images in training set: %d\n", len(trainingImages))
	fmt.Printf("Number of images in testing set: %d\n", len(testingImages))
	fmt.Printf("Number of labels in training set: %d\n", len(trainingSetLabels))
	fmt.Printf("Number of labels in training set: %d\n", len(testingSetLabels))

	imageWidth := trainingImages[0].Width
	imageHeight := trainingImages[0].Height
	network := core.CreateNetwork(imageWidth*imageHeight, 10, 1, imageWidth*imageHeight, 0.05)

	// util.PrintDense(trainingImages[0].GetDense())
	for i := 0; i < len(trainingImages); i++ {
		fmt.Printf("Image #%d, OutputCount: %d\n", i, trainingSetExpectedData[i].Len())
		err := network.TrainMonnochromeImage(trainingImages[i], trainingSetExpectedData[i])
		if err != nil {
			log.Fatal(err)
		}
	}

	// Test network on dataset to check trained accuracy
	gotRight := 0
	for i := 0; i < len(testingImages); i++ {
		result, err := network.PredictMonochromeImage(testingImages[i])
		if err != nil {
			log.Fatal(err)
		}

		// Find the index with the highest output
		maxIndex := -1
		for o := 0; o < result.Len(); i++ {
			if maxIndex == -1 {

			} else {
				if result.AtVec(maxIndex) < result.AtVec(o) {
					maxIndex = o
				}
			}
		}

		// Check if the network predicted correctly
		if maxIndex == testingSetLabels[i] {
			gotRight++
		}
	}

	fmt.Printf("Correctness: %d\n", gotRight/len(testingImages))
}

func filterAndLabelData(paths []string) (labels []int, filteredPaths []string, err error) {
	for i := 0; i < len(paths); i++ {
		curPath := paths[i] // Get current path for this loop

		fileName := filepath.Base(curPath)
		indexOfDash := strings.Index(fileName, "-") // Find where dash is in string
		numStr := fileName[3:indexOfDash]           // Extract the number substring

		num, err := strconv.Atoi(numStr) // Parse number and check for error
		if err != nil {
			return nil, nil, errors.New("Image name in invalid format: " + fileName)
		}

		// Add path to list if it's a number
		if num <= 10 {
			labels = append(labels, num-1)
			filteredPaths = append(filteredPaths, curPath)
		}
	}

	return labels, filteredPaths, nil
}

// generateExpectedOutputFromLables creates the expected output vector from what number the label is
func generateExpectedOutputFromLables(labels []int) (expectedOutputs []*mat.VecDense) {
	for i := 0; i < len(labels); i++ {
		tmp := make([]float64, 10)                                          // Create array that will hold the data temperatorly
		tmp[labels[i]] = 1                                                  // Set the corresponding numbers index to 1
		expectedOutputs = append(expectedOutputs, mat.NewVecDense(10, tmp)) // Create the vec dense from the tmp array and append it to output list
	}

	return expectedOutputs
}

func printStrArray(arr []string) {
	for i := 0; i < len(arr); i++ {
		fmt.Println(arr[i])
	}
}
