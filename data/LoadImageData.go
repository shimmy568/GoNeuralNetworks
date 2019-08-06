// Package data is the package that will handle the dataset loading the neural networks
package data

import (
	"fmt"
	"image"
	"image/png"
	"os"
)

const maxOpenFiles = 25

// LoadMonochromeImages loads multiple images into MonochromeImageData structs
func LoadMonochromeImages(paths []string) ([]*MonochromeImageData, error) {
	output := make([]*MonochromeImageData, len(paths)) // Create output data array

	quit := make(chan bool)
	errc := make(chan error)
	done := make(chan error)

	active := make(chan bool, maxOpenFiles) // Limit the number of goroutines loading files
	for i := 0; i < maxOpenFiles; i++ {     // Fill the buffered channel
		active <- true
	}

	for i := 0; i < len(paths); i++ {
		<-active // Limit the number of goroutines loading files
		go func(iter int) {
			// Load an image and check for errors
			img, err := LoadMonochromeImage(paths[iter])
			active <- true // Allow new goroutine to be created

			// Set up return chan depending on if image load errored
			ch := done
			if err != nil {
				ch = errc
			} else {
				// Load the img data into the output array if function did not error
				fmt.Printf("Loaded image: %d/%d\n", iter, len(paths))
				output[iter] = img // Copy image into output array
			}

			// Return error/done or quit
			select {
			case ch <- err:
				return
			case <-quit:
				return
			}
		}(i)
	}

	count := 0
	for {
		select {
		case err := <-errc: // If an error is thrown return now
			close(quit)
			return nil, err
		case <-done: // When a goroutine is done increment the counter until all threads are exited
			count++
			if count == len(paths) {
				return output, nil // got all N signals, so there was no error
			}
		}
	}
}

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
	src, err := png.Decode(inputFile)
	if err != nil {
		return nil, err
	}

	return src, nil
}
