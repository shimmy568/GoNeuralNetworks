package main

import (
	"errors"
	"fmt"
	"log"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shimmy568/GoNeuralNetworks/core"
	"github.com/shimmy568/GoNeuralNetworks/data"
	"github.com/shimmy568/GoNeuralNetworks/util"
	"gonum.org/v1/gonum/mat"
)

const imageWidth = 32
const imageHeight = 24
const epochCount = 50

// runHandwritingFF trains and tests the network and stuffs
func runHandwritingFF() {

	// Load and process the image data
	trainingImages, trainingLables, testingImages, testingLabels := loadAndProcessData()

	// Create the network
	n := core.CreateNetwork(imageWidth*imageHeight, 10, 1, 200, 0.05)

	// Train the network
	trainHandwritingFF(&n, trainingImages, trainingLables)

	testHandwritingFF(&n, testingImages, testingLabels)
}

// loadAndProcessData loads the data from the disk and add labels and resize and shite
func loadAndProcessData() ([]*data.MonochromeImageData, []int, []*data.MonochromeImageData, []int) {
	// Get list of images to load from disk
	paths, err := data.ListFilesDir("/home/owen/Documents/datasets/handwriting/English/Hnd/all.txt~")
	if err != nil {
		panic(err)
	}

	// Apply the prefix to paths loaded from text file
	paths = data.PrefixStringArray(paths, "/home/owen/Documents/datasets/handwriting/English/Hnd/")

	// Shuffle the data
	util.GetRand().Shuffle(len(paths), func(i, j int) { paths[i], paths[j] = paths[j], paths[i] }) // Shuffle the paths

	// Filter the images
	_, paths, err = filterAndLabelData(paths)
	if err != nil {
		log.Fatal(err)
	}

	// Segment data set into a training set and a testing set
	testingSet, trainingSet, _ := data.SegmentDataSet(paths, 0.1)

	fmt.Printf("Training Set Size: %d, Testing Set Size: %d\n", len(trainingSet), len(testingSet))

	// label the data
	trainingSetLabels, trainingSet, err := filterAndLabelData(trainingSet)
	if err != nil {
		log.Fatal(err)
	}

	testingSetLabels, testingSet, err := filterAndLabelData(testingSet)
	if err != nil {
		log.Fatal(err)
	}

	// Load images from disk
	trainingImages, err := data.LoadMonochromeImages(trainingSet, imageWidth, imageHeight)
	if err != nil {
		log.Fatal(err)
	}

	testingImages, err := data.LoadMonochromeImages(testingSet, imageWidth, imageHeight)
	if err != nil {
		log.Fatal(err)
	}

	return trainingImages, trainingSetLabels, testingImages, testingSetLabels
}

// trainHandwritingFF trains the network with the given items
func trainHandwritingFF(n *core.NeuralNet, images []*data.MonochromeImageData, labels []int) {
	expectedOutputs := generateExpectedOutputFromLables(labels)

	// Create a list that will be used to randomize the order of images that we train on per epoch
	order := make([]int, len(images))
	for i := 0; i < len(order); i++ {
		order[i] = i
	}

	// Train the network on the trainging data for the number of epochs
	for o := 0; o < epochCount; o++ {
		fmt.Printf("Epoch: %d/%d\n", o+1, epochCount) // Print the current epoch number
		for i := 0; i < len(images); i++ {
			// Shuffle order array
			util.GetRand().Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })

			// Train the network on the image given
			err := n.TrainMonnochromeImage(images[order[i]], expectedOutputs[order[i]])
			if err != nil {
				log.Fatal(err)
			}
		}
		testHandwritingFF(n, images, labels)
	}
}

// testHandwritingFF tests the network with the given data
func testHandwritingFF(n *core.NeuralNet, images []*data.MonochromeImageData, labels []int) {
	imageCount := len(images)

	// Test network on dataset to check trained accuracy
	gotRight := 0
	for i := 0; i < imageCount; i++ {
		result, err := n.PredictMonochromeImage(images[i])
		if err != nil {
			log.Fatal(err)
		}

		// Find the index with the highest output
		maxIndex := -1
		for o := 0; o < result.Len(); o++ {
			if maxIndex == -1 {
				maxIndex = o
			} else {
				if result.AtVec(maxIndex) < result.AtVec(o) {
					maxIndex = o
				}
			}
		}

		// Check if the network predicted correctly
		if maxIndex == labels[i] {
			gotRight++
		}
	}

	fmt.Printf("Got Right: %d, Out Of: %d, Ratio: %f\n", gotRight, imageCount, float64(gotRight)/float64(imageCount))
}

// generateExpectedOutputFromLables creates the expected output vector from what number the label is
func generateExpectedOutputFromLables(labels []int) (expectedOutputs []*mat.VecDense) {
	for i := 0; i < len(labels); i++ {
		tmp := make([]float64, 10) // Create array that will hold the data temperatorly
		for i := range tmp {
			tmp[i] = 0.001
		}
		tmp[labels[i]] = 0.999                                              // Set the corresponding numbers index to 0.999
		expectedOutputs = append(expectedOutputs, mat.NewVecDense(10, tmp)) // Create the vec dense from the tmp array and append it to output list
	}

	return expectedOutputs
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
