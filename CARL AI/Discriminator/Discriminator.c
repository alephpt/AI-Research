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
            weights[i][j] -= learning_rate * error[i][j] * output[i][j];
        }
    }
}

void backpropDiscriminator(Discriminator* discriminator, float discriminator_loss) {
    float** discriminator_error = calculateOutputError(discriminator->output, discriminator_loss);
    updateDiscriminatorWeights(discriminator->weights, discriminator->output, discriminator_error, discriminator->learning_rate);
    freeMatrix(discriminator_error);
}

float calculateDiscriminatorError(CNN* discriminator, float** real_data, float** fake_data, int batch_size) { 
    float error = 0; 
    
    for (int i = 0; i < batch_size; i++) { 
        float real_output = discriminator->output[i][0]; 
        float fake_output = discriminator->output[i][1]; 
        error += -(log(real_output) + log(1 - fake_output)); 
    } 
    
    return error / batch_size; 
}

void calculateGradients(CNNLayer* layer, float error) {
    float delta = error * layer->activation_derivative(layer->output);
    for (int i = 0; i < layer->n_inputs; i++) {
        for (int j = 0; j < layer->n_outputs; j++) {
            layer->weights[i][j] -= layer->learning_rate * delta * layer->input[i];
            layer->biases[j] -= layer->learning_rate * delta;
        }
    }
}

void calculateBatchDiscriminatorGradients(CNN* discriminator, float** real_data, float** fake_data, int batch_size) {
    int i, j;

    // Calculate error for real data
    for (i = 0; i < batch_size; i++) {
        for (j = 0; j < discriminator->n_outputs; j++) {
            float error = discriminator->output[i][j] - real_data[i][j];
            // Calculate gradients for each layer
            for (int k = 0; k < discriminator->n_layers; k++) {
                CNNLayer layer = discriminator->layers[k];
                calculateGradients(layer, error);
            }
        }
    }

    // Calculate error for fake data
    for (i = 0; i < batch_size; i++) {
        for (j = 0; j < discriminator->n_outputs; j++) {
            float error = discriminator->output[i][j] - fake_data[i][j];
            // Calculate gradients for each layer
            for (int k = 0; k < discriminator->n_layers; k++) {
                CNNLayer layer = discriminator->layers[k];
                calculateGradients(layer, error);
            }
        }
    }
}

void calculateDiscriminatorGradients(float* discriminator_output, float* real_data, float* fake_data, CNNLayer* layers, int n_layers) {
    for (int i = 0; i < n_layers; i++) {
        //calculate gradients for current layer
        for (int j = 0; j < layers[i].n_neurons; j++) {
            layers[i].d_weights[j] = (real_data[j] - fake_data[j]) * discriminator_output[j] * (1 - discriminator_output[j]);
            layers[i].d_biases[j] = (real_data[j] - fake_data[j]) * discriminator_output[j] * (1 - discriminator_output[j]);
        }
    }
}

void updateDiscriminatorWeights(float** weights, float** biases, CNNLayer* layers, int n_layers, float error, int batch_size) {
    float learning_rate = 0.01;
    for (int i = 0; i < n_layers; i++) {
        for (int j = 0; j < layers[i].n_weights; j++) {
            weights[i][j] -= learning_rate * layers[i].weight_gradients[j] * error / batch_size;
        }
        for (int j = 0; j < layers[i].n_biases; j++) {
            biases[i][j] -= learning_rate * layers[i].bias_gradients[j] * error / batch_size;
        }
    }
}

void trainDiscriminator(CNN* discriminator, float** real_data, float** fake_data, int batch_size) {
    // Forward pass on real data
    forwardPassCNN(real_data, discriminator->output, discriminator->n_inputs, discriminator->n_outputs, 
                    discriminator->weights, discriminator->biases, discriminator->layers, discriminator->n_layers, 
                    discriminator->filter_sizes, discriminator->n_filters, discriminator->layer_sizes);
    // Forward pass on fake data
    forwardPassCNN(fake_data, discriminator->output, discriminator->n_inputs, discriminator->n_outputs, 
                    discriminator->weights, discriminator->biases, discriminator->layers, discriminator->n_layers, 
                    discriminator->filter_sizes, discriminator->n_filters, discriminator->layer_sizes);

    // Calculate error and gradients
    float error = 0;
    for(int i = 0; i < batch_size; i++) {
        error += calculateDiscriminatorError(discriminator->output[i], real_data[i], fake_data[i]);
        calculateDiscriminatorGradients(discriminator->output[i], real_data[i], fake_data[i], discriminator->layers, 
                                        discriminator->n_layers);
    }
    error = error / batch_size;

    // Update weights and biases
    updateDiscriminatorWeights(discriminator->weights, discriminator->biases, discriminator->layers, 
                               discriminator->n_layers, error, batch_size);
}


void trainGenerator(CNN* generator, Discriminator* discriminator, int batch_size) {
    float** fake_data = generateFakeData(generator, batch_size);

    forwardPassCNN(generator->input, generator->output, generator->n_inputs, generator->n_outputs, generator->weights, 
                    generator->biases, generator->layers, generator->n_layers, generator->filter_sizes, generator->n_filters, 
                    generator->layer_sizes);

    forwardPassCNN(discriminator->input, discriminator->output, discriminator->n_inputs, discriminator->n_outputs, 
                    discriminator->weights, discriminator->biases, discriminator->layers, discriminator->n_layers, 
                    discriminator->filter_sizes, discriminator->n_filters, discriminator->layer_sizes);

    float error = calculateGeneratorError(discriminator->output, fake_data);

    calculateGeneratorGradients(generator->layers, generator->n_layers, error);

    updateGeneratorWeights(generator->weights, generator->biases, generator->layers, generator->n_layers, error, batch_size);
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