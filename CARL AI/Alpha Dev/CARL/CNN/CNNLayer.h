#pragma once
#include "Kernel.h"
#include "CNNData.h"

typedef enum PoolType {
    MAX_POOLING,
    AVG_POOLING,
    GLOBAL,
    L2_POOLING
} PoolType;

typedef enum ConvolutionType {
    STANDARD_CONVOLUTION,
    PADDED_CONVOLUTION,
    DILATION_CONVOLUTION
} ConvolutionType;

class CNNLayer {
public:
    CNNLayer(int, int);
    CNNLayer(int, int, int);
    CNNLayer(Activation, int, int);
    CNNLayer(Activation, int, int, int);
    CNNLayer(int, int, FilterDimensions);
    CNNLayer(int, int, int, FilterDimensions);
    CNNLayer(int, int, FilterDimensions, int);
    CNNLayer(int, int, int, FilterDimensions, int);
    CNNLayer(Activation, int, int, FilterDimensions);
    CNNLayer(Activation, int, int, int, FilterDimensions);
    CNNLayer(Activation, int, int, FilterDimensions, int);
    CNNLayer(Activation, int, int, int, FilterDimensions, int);

    void setStride(int);
    void convolute(ConvolutionType);

    int stride;
    int input_w;
    int input_h;

    Kernel* k;
    CNNData* data;
};