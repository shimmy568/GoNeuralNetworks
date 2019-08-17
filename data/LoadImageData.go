// Package data is the package that will handle the dataset loading the neural networks
package data

import (
	"image"
	"image/png"
	"os"

	"github.com/nfnt/resize"
)

const maxOpenFiles = 25

// LoadMonochromeImages loads multiple images into MonochromeImageData structs
// Provide a width and a height to make the loader resize the images to the given dims
func LoadMonochromeImages(paths []string, width int, height int) ([]*MonochromeImageData, error) {
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
			img, err := LoadMonochromeImage(paths[iter], width, height, active)

			// Set up return chan depending on if image load errored
			ch := done
			if err != nil {
				ch = errc
			} else {
				// Load the img data into the output array if function did not error
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
func LoadMonochromeImage(path string, width int, height int, syncChan chan bool) (*MonochromeImageData, error) {
	// Load image from file and check for error
	img, err := loadImageDataFromFile(path, width, height, syncChan)
	if err != nil {
		return nil, err
	}

	// Create the MonochromeImageData struct and add the data to it
	cols, rows := img.Bounds().Dx(), img.Bounds().Dy()
	obj := createMonochromeImageData(cols, rows)
	obj.loadImageData(img)

	//return struct
	return obj, nil
}

// loadImageDataFromFile loads an image from a file and scales it to a provided set of dimensions
// 	path: The path that the image should be loaded from
// 	width: The width that the image will be scaled to
// 	height: The height that the image will be scales to
// 	syncChan: A channel that will be sent true when the image is done loading. (used to prevent the code from having too many open files)
func loadImageDataFromFile(path string, width int, height int, syncChan chan bool) (image.Image, error) {
	// Load file and handle errors
	inputFile, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	// Decode the image file into the image object
	// Decode will figure out what type of image is in the file on its own.
	src, err := png.Decode(inputFile)
	if err != nil {
		return nil, err
	}
	inputFile.Close()
	syncChan <- true

	// Scale the image
	if width >= 0 || height >= 0 { // Check if the image needs to be scaled
		// Find new dimensions the image needs to be scaled to
		newWidth := uint(src.Bounds().Dx())
		newHeight := uint(src.Bounds().Dy())

		if width >= 0 {
			newWidth = uint(width)
		}

		if height >= 0 {
			newHeight = uint(height)
		}

		// Do the actual scaling part
		src = resize.Resize(newWidth, newHeight, src, resize.Lanczos3)
	}

	return src, nil
}
