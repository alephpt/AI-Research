#include "CNNLayer.h"
#include "../utilities/shop.h"

CNNLayer* initializeCNNLayer(int n_inputs, int n_outputs, ActivationFunction activation_function) {
    CNNLayer* layer = (CNNLayer*)malloc(sizeof(CNNLayer));
    layer->n_inputs = n_inputs;
    layer->n_outputs = n_outputs;
    layer->weights = allocateMatrix(n_inputs, n_outputs);
    layer->biases = allocateVector(n_outputs);
    layer->activation_function = activation_function;
    return layer;
}