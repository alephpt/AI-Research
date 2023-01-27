#include "CNNLayer.h"
#include "../utilities/shop.h"

float activation_derivative(CNNLayer* layer, float output) {
    if (layer->activation_type == SIGMOID) {
        return output * (1 - output);
    } else if (layer->activation_type == TANH) {
        return 1 - (output * output);
    } else if (layer->activation_type == RELU) {
        return (output > 0) ? 1 : 0;
    } else if (layer->activation_type == LEAKY_RELU) {
        return (output > 0) ? 1 : 0.01;
    } else {
        printf("Invalid activation type\n");
        return 0;
    }
}

CNNLayer* initializeCNNLayer(int n_inputs, int n_outputs, ActivationFunction activation_function) {
    CNNLayer* layer = (CNNLayer*)malloc(sizeof(CNNLayer));
    layer->n_inputs = n_inputs;
    layer->n_outputs = n_outputs;
    layer->activation_derivative = activation_derivative(CNNLayer* layer, float output);
    layer->weights = allocateMatrix(n_inputs, n_outputs);
    layer->biases = allocateVector(n_outputs);
    layer->activation_function = activation_function;
    return layer;
}

