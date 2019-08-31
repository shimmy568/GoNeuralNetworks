package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/shimmy568/GoNeuralNetworks/core"
)

// This file is for holding the logic associated with having

func runMnistDataFF() {
	n := core.CreateNetwork(28*28, 10, 1, 200, 0.1)

	mnistTrain(&n)
	mnistPredict(&n)
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
