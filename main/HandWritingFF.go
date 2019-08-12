package main

import (
	"errors"
	"fmt"
	"log"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shimmy568/NGin/data"
)

// This file is for holding the logic associated with having

func runHandwritingFF() {
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

	// Load images from disk
	trainingImages, err := data.LoadMonochromeImages(trainingSet, 64, 64)
	if err != nil {
		log.Fatal(err)
	}

	testingImages, err := data.LoadMonochromeImages(testingSet, 64, 64)
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

	/* // Train network for the dataset
	imageWidth := trainingImages[0].Width
	imageHeight := trainingImages[0].Height
	core.CreateNetwork(imageHeight*imageHeight, 10, 1, imageHeight*imageWidth, 0.05)
	for i := 0; i < len(trainingImages); i++ {
		// TODO
		// n.TrainMonnochromeImage(trainingImages[i])
	} */

	// Test network on dataset to check trained accuracy
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

func printStrArray(arr []string) {
	for i := 0; i < len(arr); i++ {
		fmt.Println(arr[i])
	}
}
