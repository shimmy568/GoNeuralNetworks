package main

import (
	"fmt"
	"log"

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
	trainingSet, testingSet, _ := data.SegmentDataSet(paths, 0.5)

	// Load images from disk
	trainingImages, err := data.LoadMonochromeImages(trainingSet)
	if err != nil {
		log.Fatal(err)
	}

	testingImages, err := data.LoadMonochromeImages(testingSet)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Number of images in training set: %d\n", len(trainingImages))
	fmt.Printf("Number of images in testing set: %d\n", len(testingImages))

	// Train network for the dataset

	// Test network on dataset to check trained accuracy
}

func printStrArray(arr []string) {
	for i := 0; i < len(arr); i++ {
		fmt.Println(arr[i])
	}
}
