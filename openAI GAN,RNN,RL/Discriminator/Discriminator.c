#include "Discriminator.h"

float computeGeneratorLoss(float* discriminator_output, int batch_size) {
    float loss = 0.0;
    for (int i = 0; i < batch_size; i++) {
        loss += log(1 - discriminator_output[i]);
    }
    return -loss / batch_size;
}

Discriminator* createDiscriminator(int n_inputs, int n_outputs, int n_layers, int* layer_sizes) {
    Discriminator* discriminator = (Discriminator*)malloc(sizeof(Discriminator));
    //    discriminator->cnn = createCNN(n_inputs, n_outputs, n_layers, filter_sizes, n_filters, layer_sizes); 
    discriminator->n_inputs = n_inputs;
    discriminator->n_outputs = n_outputs;
    discriminator->n_layers = n_layers;
    discriminator->layer_sizes = layer_sizes;
    discriminator->weights = initWeights(n_layers, layer_sizes);
    discriminator->biases = initBiases(n_layers, layer_sizes);
    return discriminator;
}

float computeDiscriminatorLoss(float** discriminator_output, float** real_data, int batch_size) {
    float loss = 0;
    for (int i = 0; i < batch_size; i++) {
        loss += log(discriminator_output[i][0]) * real_data[i][0] + log(1 - discriminator_output[i][1]) * (1 - real_data[i][0]);
    }
    return -loss / batch_size;
}