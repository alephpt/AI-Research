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

void updateDiscriminatorOutput(Discriminator* discriminator, float** output_error) {
    // Implement backpropagation algorithm to update the discriminator->output
    // ...
    // Update the output_error based on the error and the learning rate
    for (int i = 0; i < discriminator->n_outputs; i++) {
        for (int j = 0; j < discriminator->batch_size; j++) {
            discriminator->output[i][j] -= discriminator->learning_rate * output_error[i][j];
        }
    }
}

static float** calculateOutputError(float** output, float discriminator_loss) {
    int n_outputs = sizeof(output) / sizeof(output[0]);
    float** output_error = allocateMatrix(1, n_outputs);
    for (int i = 0; i < n_outputs; i++) {
        output_error[0][i] = output[0][i] - discriminator_loss;
    }
    return output_error;
}

void updateDiscriminatorWeights(float** weights, float** output, float** error, float learning_rate) {
    int n_outputs = sizeof(output[0]);
    int n_inputs = sizeof(weights) / sizeof(weights[0]);
    for (int i = 0; i < n_inputs; i++) {
        for (int j = 0; j < n_outputs; j++) {
            weights[i][j] -= learning_rate * error[j] * output[i];
        }
    }
}

void backpropDiscriminator(Discriminator* discriminator, float discriminator_loss) {
    float** discriminator_error = calculateOutputError(discriminator->output, discriminator_loss);
    updateDiscriminatorWeights(discriminator->weights, discriminator->output, discriminator_error, discriminator->learning_rate);
    freeMatrix(discriminator_error);
}

void syncDiscriminatorToCNN(CNN* generator, Discriminator* discriminator) {
    discriminator->input = generator->input;
    discriminator->output = generator->output;
    discriminator->n_inputs = generator->n_inputs;
    discriminator->n_outputs = generator->n_outputs;
    discriminator->n_layers = generator->n_layers;
    discriminator->layers = generator->layers;
    discriminator->layer_sizes = generator->layer_sizes;
    discriminator->weights = generator->weights;
    discriminator->biases = generator->biases;
}