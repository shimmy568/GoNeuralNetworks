package main

import (
	"fmt"

	"github.com/shimmy568/NGin/pkg/core"
)

func main() {
	net := core.CreateNetwork(3, 10, 1, 5, 0.1)

	data := []float64{1, 2, 3}
	o := net.ForwardPass(data)

	fmt.Printf("%v\n", o)
}
