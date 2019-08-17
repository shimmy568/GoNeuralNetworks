package data

import (
	"image"
	"image/color"

	"gonum.org/v1/gonum/mat"
)

// This file is responsable for handling things related to the MonochromeImageData struct

// MonochromeImageData is a struct that holds data for a monochrome image
type MonochromeImageData struct {
	// Width and Height are for storing the dims of the image
	Width, Height int

	// data is the 2D matrix that stores the image data
	data *image.Gray
}

// createMonochromeImageData creates a blank new MonochromeImageData struct
func createMonochromeImageData(cols, rows int) *MonochromeImageData {
	// Create the struct and initialize the static fields
	obj := &MonochromeImageData{
		Width:  cols,
		Height: rows,
	}

	// Set up the data field
	rect := image.Rect(0, 0, cols, rows)
	obj.data = image.NewGray(rect)

	return obj
}

// loadImageData takes an image and loads it's data into the MonochromeImageData
func (m *MonochromeImageData) loadImageData(img image.Image) {
	// Loop through all pixels in image and convert to grayscale
	for col := 0; col < m.Width; col++ {
		for row := 0; row < m.Height; row++ {
			// Convert color to gray and update to struct data
			oldColor := img.At(col, row)
			grayColor := color.GrayModel.Convert(oldColor)
			m.data.Set(col, row, grayColor)
		}
	}
}

// GetDense returns a matrix representation of the internal data
func (m *MonochromeImageData) GetDense() *mat.Dense {
	// Create the mat for the data to be entered into
	data := mat.NewDense(m.Height, m.Width, nil)

	// Loop through all the pixels in the image
	for col := 0; col < m.Width; col++ {
		for row := 0; row < m.Height; row++ {
			// Convert the pixel color data to a brightness level and return it
			pixelColor := m.data.At(col, row)
			data.Set(row, col, getPixelBrightnessLevel(pixelColor))
		}
	}

	return data
}

// getPixelBrightnessLevel converts a color to a brightness value
func getPixelBrightnessLevel(c color.Color) float64 {
	r, g, b, _ := c.RGBA()
	return float64(r+g+b) / 3
}
