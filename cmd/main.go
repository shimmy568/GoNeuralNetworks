package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func main() {
	u := mat.NewVecDense(3, []float64{1, 2, 3})
	v := mat.NewVecDense(3, []float64{4, 5, 6})

	// This doesn’t work
	// a := u[0]

	// Find 1st element of u (returns value)
	u.AtVec(1)

	// Equivalent getter method for vectors
	// The At method may be used on anything that satisfies the Matrix // interface.
	// (returns value)
	u.At(1, 0)

	// Overwrite 1st element of u with 33.2
	u.SetVec(1, 33.2)

	// All vector addition is done in place to reduce overhead
	w := mat.NewVecDense(3, nil)
	w.AddVec(u, v) // Add u and v into w
	// u.AddVec(u, v) // Add u and v into u (to save space)
	println("u + v: ")
	matPrint(w)

	// Add scaled vector
	// u + alpha * v
	// u + 2v
	w.AddScaledVec(u, 2, v)
	println("u + 2 * v: ")
	matPrint(w)

	// Subtract v from u
	// v - u
	w.SubVec(u, v)
	println("v - u: ")
	matPrint(w)

	// Scale u by alpha
	// Just multiply a vector by a scalar
	w.ScaleVec(23, u)
	println("u * 23: ")
	matPrint(w)

	// Compute the dot product of u and v
	// Since float64’s don’t have a dot method, this is not done
	//inplace
	d := mat.Dot(u, v)
	println("u dot v: ", d)

	// Find length of v
	l := v.Len()
	println("Length of v: ", l)
}
