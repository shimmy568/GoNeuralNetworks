// Package data is the package that will handle the dataset loading the neural networks
package data

import (
	"image"
	"os"
)

// This file is responsable for loading the image data into an image struct

// LoadMonochromeImage loads an image into a MonochromeImageData given a path to the image
func LoadMonochromeImage(path string) (*MonochromeImageData, error) {
	// Load image from file and check for error
	img, err := loadImageDataFromFile(path)
	if err != nil {
		return nil, err
	}

	// Create the MonochromeImageData struct and add the data to it
	bounds := img.Bounds()
	rows, cols := bounds.Max.X, bounds.Max.Y
	obj := createMonochromeImageData(rows, cols)
	obj.loadImageData(img)

	//return struct
	return obj, nil
}

func loadImageDataFromFile(path string) (image.Image, error) {
	// Load file and handle errors
	inputFile, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer inputFile.Close()

	// Decode the image file into the image object
	// Decode will figure out what type of image is in the file on its own.
	src, _, err := image.Decode(inputFile)
	if err != nil {
		return nil, err
	}

	return src, nil
}
