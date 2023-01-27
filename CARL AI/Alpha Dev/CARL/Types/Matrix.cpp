#include "Matrix.h"

Matrix* initMatrix(int rows, int cols) {
    Matrix* newMatrix = new Matrix;
    newMatrix->rows = rows;
    newMatrix->cols = cols;
    newMatrix->data = new float[rows * cols];
    return newMatrix;
}