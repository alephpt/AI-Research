#pragma once

typedef struct Vector{
    float* data;
    int size;
} Vector;

Vector* initVector(int size);

