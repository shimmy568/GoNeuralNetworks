// Package util will be used to store generic helper functions
package util

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

var r *rand.Rand

// GetRand returns a global pseudo random number generator
func GetRand() *rand.Rand {
	if r == nil {
		r = rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	}
	return r
}

// PrintDense prints a dense to the console in a human readable format
func PrintDense(data *mat.Dense) {
	f := mat.Formatted(data, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("mat:\na = % v\n\n", f)
}
