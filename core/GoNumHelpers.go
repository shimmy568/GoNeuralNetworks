package core

import "gonum.org/v1/gonum/mat"

// dot is a wrapper function for the gonum dot product functionality
func dot(m, n mat.Matrix) *mat.Dense {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

// apply is a wrapper function for the application of a function to a matrix
func apply(fn func(i, j int, v float64) float64, m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

// multiply is a wrapper function for an element wise multiplication of two matrcies
func multiply(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

// subtract subtracts a two matrcies from each other
func subtract(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}
