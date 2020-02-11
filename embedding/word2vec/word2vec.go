//package word2vec provides functionality to import and
// operate on word2vec generated model.
// Ref. https://github.com/danieldk/go2vec/blob/master/go2vec.go
package word2vec

import (
	"fmt"

	"github.com/gonum/blas"
	// cblas "github.com/gonum/blas/cgo"
)

type Embedding struct {
	blas      blas.Float32Level2
	matrix    []float32
	embedSize int
	indices   map[string]int
	words     []string
}

func ReadWord2VecBinary() interface{} {
	fmt.Println("Reading word2vec file...")

	return nil
}
