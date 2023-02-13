#pragma once
#include "../Types/Types.h"
#include "Kernel.h"

typedef enum {
    CNN_CONVOLUTION_LAYER,
    CNN_POOLING_LAYER,
    CNN_FLATTERNING_LAYER,
    CNN_NEURAL_NETWORK
} CNNLayerType;

typedef struct CNNLayer {
    int n_kernels = 0;
    vector<Kernel*> kernels;
    CNNLayerType layer_type = CNN_CONVOLUTION_LAYER;
} CNNLayer;