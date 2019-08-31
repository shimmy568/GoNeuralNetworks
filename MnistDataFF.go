package main

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/shimmy568/GoNeuralNetworks/core"
)

// This file is for holding the logic associated with having

func mnistDataFF() {
	n := core.CreateNetwork(28*28, 10, 1, 200, 0.1)

	mnistTrain(&n)
	mnistPredict(&n)

	// OLD CODE
	// w, h := trainingImages[0].GetDense().Dims()
	// fmt.Printf("Width of training images: %d,%d\n", w, h)

	// w, h = testingImages[0].GetDense().Dims()
	// fmt.Printf("Width of testing images: %d,%d\n", w, h)

	// fmt.Printf("Number of images in training set: %d\n", len(trainingImages))
	// fmt.Printf("Number of images in testing set: %d\n", len(testingImages))
	// fmt.Printf("Number of labels in training set: %d\n", len(trainingSetLabels))
	// fmt.Printf("Number of labels in training set: %d\n", len(testingSetLabels))

	// imageWidth := trainingImages[0].Width
	// imageHeight := trainingImages[0].Height
	// network := core.CreateNetwork(imageWidth*imageHeight, 10, 1, 200, 1)

	// epochCount := 10
	// for o := 0; o < epochCount; o++ {
	// 	fmt.Printf("Iteration: %d/%d\n", o+1, epochCount)
	// 	for i := 0; i < len(trainingImages); i++ {
	// 		//fmt.Printf("Image #%d/%d, OutputCount: %d\n", i+1, len(trainingImages), trainingSetExpectedData[i].Len())
	// 		err := network.TrainMonnochromeImage(trainingImages[i], trainingSetExpectedData[i])
	// 		if err != nil {
	// 			log.Fatal(err)
	// 		}
	// 	}
	// }

	// // Test network on dataset to check trained accuracy
	// gotRight := 0
	// for i := 0; i < len(testingImages); i++ {
	// 	fmt.Printf("Testing image %d/%d\n", i+1, len(testingImages))
	// 	result, err := network.PredictMonochromeImage(testingImages[i])
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}

	// 	// Find the index with the highest output
	// 	maxIndex := -1
	// 	for o := 0; o < result.Len(); o++ {
	// 		if maxIndex == -1 {
	// 			maxIndex = o
	// 		} else {
	// 			if result.AtVec(maxIndex) < result.AtVec(o) {
	// 				maxIndex = o
	// 			}
	// 		}
	// 	}

	// 	// Check if the network predicted correctly
	// 	fmt.Printf("Result: %d, Expected: %d\n", maxIndex, testingSetLabels[i])
	// 	if maxIndex == testingSetLabels[i] {
	// 		gotRight++
	// 		fmt.Println("Got it right!!")
	// 	}
	// }

	// fmt.Printf("Got Right: %d, Out Of: %d, Ratio: %f\n", gotRight, len(testingImages), float64(gotRight)/float64(len(testingImages)))
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

	for epochs := 0; epochs < 1; epochs++ {
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