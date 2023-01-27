#include "Tensor.h"

int getRank(Tensor* t) { return t->rank; }

Tensor* initTensor(int rank, int* dims) {
    Tensor* t = new Tensor;

    t->rank = rank;

    t->shape = new int[rank];

    for (int i = 0; i < rank; i++) {
        t->shape[i] = dims[i];
    }
    
    int total_elements = 1;
    for (int i = 0; i < rank; i++) {
        total_elements *= dims[i];
    }

    t->data = (float*)malloc(total_elements * sizeof(float));

    return t;
}