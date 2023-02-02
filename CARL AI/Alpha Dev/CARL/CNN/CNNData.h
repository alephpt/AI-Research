#pragma once
#include <vector>
#include "Kernel.h"
#include "../types/Activation.h"

typedef enum {
    POOL_LAYER,
    CONVOLUTION_LAYER,
} LayerType;

typedef struct CNNFeature {
    int index = 0;
    int width = 0;
    int height = 0;
    FilterStyle filter_style;
    std::vector<std::vector<float>> filter;
    std::vector<std::vector<float>> values;
} CNNFeature;

typedef struct CNNSample {
    int n_features = 0;
    LayerType layer;
    FilterDimensions kernel_dimensions;
    Activation activation_type;
    std::vector<CNNFeature*> features;
} CNNSample;

typedef struct CNNData {
    CNNFeature input;
    CNNFeature output;
    std::vector<CNNSample> layers;
    int n_layers = 0;
} CNNData;