#pragma once
#include "Kernel.h"

class Convolution {
public:
    Convolution(int, int, FilterDimensions);

    std::vector<std::vector<float>> convolute(std::vector<std::vector<float>>, int, int, int*, int*);

    int stride;
    int input_w;
    int input_h;
    int padding_x;
    int padding_y;
    Kernel* k;
};