// Package util will be used to store generic helper functions
package util

import (
	"math/rand"
	"time"
)

var r *rand.Rand

// GetRand returns a global pseudo random number generator
func GetRand() *rand.Rand {
	if r == nil {
		r = rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	}
	return r
}
