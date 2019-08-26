package main

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/shimmy568/GoNeuralNetworks/util"

	"gonum.org/v1/gonum/mat"

	"github.com/shimmy568/GoNeuralNetworks/core"
	"github.com/shimmy568/GoNeuralNetworks/data"
)

// This file is for holding the logic associated with having

func runHandwritingFF() {
	n := core.CreateNetwork(28*28, 10, 1, 100, 0.1)
	mnistTrain(&n)
	mnistPredict(&n)

	return
	log.SetFlags(log.LstdFlags | log.Lshortfile)

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

	// Generate the expected vectors for each label for testing and training
	trainingSetExpectedData := generateExpectedOutputFromLables(trainingSetLabels)
	// testingSetExpectedData := generateExpectedOutputFromLables(testingSetLabels)

	// Load images from disk
	trainingImages, err := data.LoadMonochromeImages(trainingSet, 32, 24)
	if err != nil {
		log.Fatal(err)
	}

	testingImages, err := data.LoadMonochromeImages(testingSet, 32, 24)
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

	imageWidth := trainingImages[0].Width
	imageHeight := trainingImages[0].Height
	network := core.CreateNetwork(imageWidth*imageHeight, 10, 1, 200, 1)

	epochCount := 10
	for o := 0; o < epochCount; o++ {
		fmt.Printf("Iteration: %d/%d\n", o+1, epochCount)
		for i := 0; i < len(trainingImages); i++ {
			//fmt.Printf("Image #%d/%d, OutputCount: %d\n", i+1, len(trainingImages), trainingSetExpectedData[i].Len())
			err := network.TrainMonnochromeImage(trainingImages[i], trainingSetExpectedData[i])
			if err != nil {
				log.Fatal(err)
			}
		}
	}

	// Test network on dataset to check trained accuracy
	gotRight := 0
	for i := 0; i < len(testingImages); i++ {
		fmt.Printf("Testing image %d/%d\n", i+1, len(testingImages))
		result, err := network.PredictMonochromeImage(testingImages[i])
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
		fmt.Printf("Result: %d, Expected: %d\n", maxIndex, testingSetLabels[i])
		if maxIndex == testingSetLabels[i] {
			gotRight++
			fmt.Println("Got it right!!")
		}
	}

	fmt.Printf("Got Right: %d, Out Of: %d, Ratio: %f\n", gotRight, len(testingImages), float64(gotRight)/float64(len(testingImages)))
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

func printStrArray(arr []string) {
	for i := 0; i < len(arr); i++ {
		fmt.Println(arr[i])
	}
}

func mnistTrain(net *core.NeuralNet) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()

	for epochs := 0; epochs < 5; epochs++ {
		fmt.Printf("Epoch #%d\n", epochs)
		testFile, _ := os.Open("mnist_dataset/mnist_train.csv")
		r := csv.NewReader(bufio.NewReader(testFile))
		i := 0
		for {
			i++
			if i%1000 == 0 {
				fmt.Printf("Item: %d\n", i)
			}
			record, err := r.Read()
			if err == io.EOF {
				break
			}

			inputs := mat.NewVecDense(net.GetInputCount(), nil)
			for i := 0; i < net.GetInputCount(); i++ {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs.SetVec(i, (x/255.0*0.999)+0.001)
			}

			targets := mat.NewVecDense(10, nil)
			for i := 0; i < 10; i++ {
				targets.SetVec(i, 0.001)
			}
			x, _ := strconv.Atoi(record[0])
			targets.SetVec(x, 0.999)

			item := core.CreateTrainingItem(inputs, targets)
			net.Train(item)
		}
		testFile.Close()
	}
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train: %s\n", elapsed)
}

func mnistPredict(net *core.NeuralNet) {
	t1 := time.Now()
	checkFile, _ := os.Open("mnist_dataset/mnist_test.csv")
	defer checkFile.Close()

	score := 0
	r := csv.NewReader(bufio.NewReader(checkFile))
	count := 0
	for {
		count++
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, net.GetInputCount())
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255.0 * 0.999) + 0.001
		}
		outputs := net.Predict(inputs)
		best := 0
		highest := 0.0

		for i := 0; i < net.GetOutputCount(); i++ {
			if outputs.AtVec(i) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}
		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}
	}

	elapsed := time.Since(t1)
	fmt.Printf("Time taken to check: %s\n", elapsed)
	fmt.Printf("score: %d, out of: %d, ratio: %f\n", score, count, float64(score)/float64(count))
}
