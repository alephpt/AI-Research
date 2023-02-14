#pragma once
#include "../Types/Types.h"
#include "Kernel.h"

const std::string CNNLayerStrings[] = { "Convolutional Layer", "Pooling Layer", "Flattening Layer", "Fully Connected Neural Network" };

typedef enum {
    CNN_CONVOLUTION_LAYER,
    CNN_POOLING_LAYER,
    CNN_FLATTENING_LAYER,
    CNN_FULLY_CONNECTED
} CNNLayerType;

typedef struct CNNLayer {
    int n_kernels = 0;
    union {
        vector<Kernel*> kernels;

    }
    CNNLayerType layer_type;
} CNNLayer;