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

class Kernel{
    Kernel(FilterDimensions);
    Kernel(FilterDimensions, int);
    ~Kernel();
    std::vector<std::vector<float>> values;
    int rows, columns; 
};
