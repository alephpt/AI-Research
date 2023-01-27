#pragma once
#include <stdlib.h>

typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix;

Matrix* initMatrix(int rows, int cols);
