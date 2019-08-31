package main

import (
	"fmt"
	"log"

	"github.com/shimmy568/GoNeuralNetworks/core"
	"github.com/shimmy568/GoNeuralNetworks/data"
	"github.com/shimmy568/GoNeuralNetworks/util"
)

// runHandwritingFF trains and tests the network and stuffs
func runHandwritingFF() {

	// Load and process the image data
	trainingImages, trainingLables, testingImages, testingLabels := loadAndProcessData()

	n := core.CreateNetwork()
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
	testingSet, trainingSet, _ := data.SegmentDataSet(paths, 5)

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
	trainingImages, err := data.LoadMonochromeImages(trainingSet, 32, 24)
	if err != nil {
		log.Fatal(err)
	}

	testingImages, err := data.LoadMonochromeImages(testingSet, 32, 24)
	if err != nil {
		log.Fatal(err)
	}

	return trainingImages, trainingSetLabels, testingImages, testingSetLabels
}

// trainHandwritingFF trains the network with the given items
func trainHandwritingFF(n *core.NeuralNet) {

}

// testHandwritingFF tests the network with the given data
func testHandwritingFF(n *core.NeuralNet) {
	// TODO test the network
}
