#include "Matrix.h"
#include <stdio.h>

Matrix* initMatrix(int rows, int cols) {
    Matrix* newMatrix = new Matrix;
    newMatrix->rows = rows;
    newMatrix->cols = cols;
    newMatrix->data = new float[rows * cols];
    return newMatrix;
}

void printMatrix(float** data, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f", data[i][j]);
        }
    }
}
