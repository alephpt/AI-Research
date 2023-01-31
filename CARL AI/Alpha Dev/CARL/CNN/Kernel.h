#pragma once
#include <vector>
#include <string>

const std::string filterString[] = { "ONExONE", "ONExTHREE", "ONExN", "THREExONE", "NxONE", "THREExTHREE", "FIVExFIVE" };

typedef enum FilterDimensions {
    ONExONE,
    ONExTHREE,
    ONExN,
    THREExONE,
    NxONE,
    THREExTHREE,
    FIVExFIVE
} FilterDimensions;

typedef struct Kernel{
    std::vector<std::vector<float>> values;
    int rows, columns; 
} Kernel;

Kernel* initKernel(FilterDimensions);
Kernel* initKernel(FilterDimensions, int n);