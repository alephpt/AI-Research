#pragma once
#include <stdlib.h>
#include "Matrix.h"

typedef std::vector<matrix> tensor;

typedef struct {
	int rank;
	int* shape;
	float* data;
} Tensor;

Tensor* initTensor(int rank, int* dims);
// addTensors
// subtractTensors
// scalarMultiply
// scalarDivide
// matrixMultiply
// dotProduct
// tensorProduct
// innerProduct
// outerProduct
// transpose
// broadcast
// determinant
// inverse
// trace
// contraction
// reshape
// slice