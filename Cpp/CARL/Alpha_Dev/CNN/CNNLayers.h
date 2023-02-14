#pragma once
#include "../Types/Types.h"
#include "Kernel.h"
#include "Pool.h"
#include <variant>

const std::string CNNLayerStrings[] = { "Convolutional Layer", "Pooling Layer", "Flattening Layer", "Fully Connected Neural Network" };

typedef enum {
    CNN_CONVOLUTION_LAYER,
    CNN_POOLING_LAYER,
    CNN_FLATTENING_LAYER,
    CNN_FULLY_CONNECTED
} CNNLayerType;

typedef struct ConvolutionLayer {
    int n_kernels;
    vector<Kernel*> kernels;
} ConvolutionLayer;

typedef std::variant<ConvolutionLayer*, PoolingLayer*> LayerData;

typedef struct CNNLayer {
    LayerData data;
    CNNLayerType type;
    CNNLayer(){};
} CNNLayer;


inline ConvolutionLayer* getConvolutionLayer(LayerData data) { return std::get<ConvolutionLayer*>(data); }
inline PoolingLayer* getPoolingLayer(LayerData data) { return std::get<PoolingLayer*>(data); }