#include "Vector.h"

Vector* initVector(int size) {
    Vector* newVector = new Vector;
    newVector->size = size;
    newVector->data = new float[size];
    return newVector;
}