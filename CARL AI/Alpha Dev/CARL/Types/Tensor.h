#pragma once
#include <stdlib.h>

typedef struct {
	int rank;
	int* shape;
	float* data;
} Tensor;

Tensor* initTensor(int rank, int* dims);
int getRank(Tensor);

