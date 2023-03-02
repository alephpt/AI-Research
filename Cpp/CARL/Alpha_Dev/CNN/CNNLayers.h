#pragma once
#include "../Types/Types.h"
#include "Convolution.h"
#include "Pool.h"
#include <variant>

const std::string CNNLayerStrings[] = { "Convolutional Layer", "Pooling Layer", "Flattening Layer", "Fully Connected Neural Network" };

typedef enum {
    CNN_CONVOLUTION_LAYER,
    CNN_POOLING_LAYER,
    CNN_FLATTENING_LAYER,
    CNN_FULLY_CONNECTED_LAYER
} CNNLayerType;

typedef std::variant<Convolution*, Pool*> LayerData;

typedef struct CNNLayer {
    CNNLayer(CNNLayerType layer_type) {
        type = layer_type;
        switch (layer_type) {
            case CNN_CONVOLUTION_LAYER: {
                data = new Convolution;
                break;
            }
            case CNN_POOLING_LAYER: {
                data = new Pool;
                break;
            }
            case CNN_FLATTENING_LAYER: {
                break;
            }
            case CNN_FULLY_CONNECTED_LAYER: {
                break;
            }
        }
    };

    LayerData data;
    CNNLayerType type;
} CNNLayer;
