#pragma once
#include <vector.h>

typedef struct Kernel {
    std::vector<std::vector<float>> values;
    int size; 
} Kernel;