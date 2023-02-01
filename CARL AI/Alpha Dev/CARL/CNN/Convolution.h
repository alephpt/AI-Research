#pragma once
#include "Kernel.h"

class Convolution {
public:
    Convolution(int, int);
    Convolution(Activation, int, int);
    Convolution(int, int, FilterDimensions);
    Convolution(Activation, int, int, FilterDimensions);

    std::vector<std::vector<float>> convolute(std::vector<std::vector<float>>, int, int, int*, int*);
    std::vector<std::vector<float>> paddedConvolute(std::vector<std::vector<float>>, int, int, int*, int*);
    std::vector<std::vector<float>> dilationConvolute(std::vector<std::vector<float>>, int, int, int*, int*);
    void setStride(int);

    int stride;
    int input_w;
    int input_h;
    Kernel* k;
};