package main

import (
	"fmt"

	"github.com/shimmy568/NGin/data"
)

// This file is for holding the logic associated with having

func runHandwritingFF() {
	// TODO Implement pseudo code

	// Get list of images to load from disk
	paths, err := data.ListFilesDir("~/Documents/datasets/handwriting/English/Hnd/all.txt~")
	if err != nil {
		panic(err)
	}

	// Segment data set into a training set and a testing set
	trainingSet, testingSet, _ := data.SegmentDataSet(paths, 0.5)

	printStrArray(trainingSet)
	fmt.Println("---------")
	printStrArray(testingSet)

	// Load images from disk

	// Train network for the dataset

	// Test network on dataset to check trained accuracy
}

func printStrArray(arr []string) {
	for i := 0; i < len(arr); i++ {
		fmt.Println(arr[i])
	}
}
