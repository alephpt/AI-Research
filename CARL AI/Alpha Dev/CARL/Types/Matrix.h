#pragma once
#include <stdlib.h>

typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix;

Matrix* initMatrix(int rows, int cols);
void printMatrix(float** data, int rows, int cols);
// add
// subtract
// scalarMultiply
// scalarDivide
// matrixMultiply
// matrixExponent
// matrixLogarithm
// matrixPower
// transpose
// determinant
// inverse
// trace
// rank
// eigenvalue
// eigenvector
// orthogonal
// unitary
// LU
// QR