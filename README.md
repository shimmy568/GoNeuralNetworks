# GoNeuralNetworks

*Still Major Work in Progress*

Golang Neural Networks Stuffs

## Packages

- core: The core logic and data structures for the neural networks (training, predictions, ...)
- data: The logic and data structures for loading and processing the data for the network (loading images from disk, turning image into matricies, ...)
- util: Super general utilities (printing \*mat.Dense, getting random number generator, ...)
- main: The actual code that sets up, trains, and tests the networks using the other packages

## Cool Stuff

- Images are loaded from disk and pre-processed in parrall using goroutines
- Implemented from scratch using GoNum
- That's about it. This is still just me making an artificial neural network in golang and that's about it

## Dependencies

- GoNum: https://github.com/gonum/gonum
- Image Resizing: https://github.com/nfnt/resize

## Refrences

- https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/
- Shit load of stuff from https://stackoverflow.com/
