#include "CNN.h"
#include "../utilities/shop.h"

const float LEARNING_RATE = 0.001;

CNN* createCNN(int n_inputs, int n_outputs, int n_layers, int* filter_sizes, int n_filters, int* layer_sizes) {
    CNN* cnn = (CNN*)malloc(sizeof(CNN));
    cnn->n_inputs = n_inputs;
    cnn->n_outputs = n_outputs;
    cnn->n_layers = n_layers;
    cnn->filter_sizes = filter_sizes;
    cnn->n_filters = n_filters;
    cnn->layer_sizes = layer_sizes;
    cnn->weights = (float**)malloc(sizeof(float*) * n_layers);
    cnn->biases = (float**)malloc(sizeof(float*) * n_layers);

    for (int i = 0; i < n_layers; i++) {
        cnn->weights[i] = allocate2DArray(filter_sizes[i], filter_sizes[i], n_filters);
        cnn->biases[i] = allocate1DArray(n_filters);
    }
    return cnn;
}

void destroyCNN(CNN* cnn) {
    int i;
    for (i = 0; i < cnn->n_layers; i++) {
        free(cnn->weights[i]);
        free(cnn->biases[i]);
    }
    free(cnn->weights);
    free(cnn->biases);
    free(cnn->filter_sizes);
    free(cnn->layer_sizes);
    free(cnn);
}

int predictCNN(CNN* cnn, float* testData) {
    float* output = forwardPropCNN_A(cnn, testData);
    int prediction = 0;
    float max_value = output[0];
    for (int i = 1; i < cnn->n_outputs; i++) {
        if (output[i] > max_value) {
            max_value = output[i];
            prediction = i;
        }
    }
    return prediction;
}


float evaluateCNN(CNN* cnn, float** testData, int* testLabels) {
    int n_correct = 0;
    int n_total = 0;
    for (int i = 0; i < sample_size; i++) {
        int prediction = predictCNN(cnn, testData[i]);
        if (prediction == testLabels[i]) {
            n_correct++;
        }
        n_total++;
    }
    return (float)n_correct / n_total;
}

void updateGeneratorOutput(CNN* generator, float** error) {
    for (int i = 0; i < generator->n_outputs; i++) {
        for (int j = 0; j < generator->batch_size; j++) {
            generator->output[i][j] -= error[i][j];
        }
    }
}

void forwardPropCNN_B(CNN* cnn, float* testData) {
    float* input = testData;
    float* layer;
    for (int i = 0; i < cnn->n_layers; i++) {
        float* dotProductResult = dotProduct(cnn->weights[i], input, cnn->n_inputs);
        float* bias = cnn->biases[i];
        add1DArrayLayer(dotProductResult, bias, cnn->layer_sizes[i], layer);
        sigmoidLayer(layer, cnn->layer_sizes[i]);
        input = layer;
    }
    cnn->output = input;
}

float evaluateCNNwithSS(CNN* cnn, float** testData, float* testLabels, int sample_size) {
    int correct = 0;
    for (int i = 0; i < sample_size; i++) {
        float* output = forwardPropCNN_B(cnn, testData[i]);
        int maxIndex = 0;
        for (int j = 1; j < cnn->n_outputs; j++) {
            if (output[j] > output[maxIndex]) {
                maxIndex = j;
            }
        }
        if (testLabels[i] == maxIndex) {
            correct++;
        }
    }
    return (float)correct / sample_size;
}

float** forwardPropCNNdiscriminator(Discriminator* discriminator, float** generator_output) {
    int i, j, k;
    int n_samples = sizeof(generator_output) / sizeof(generator_output[0]);
    int n_outputs = discriminator->n_outputs;

    // Perform the matrix multiplication
    float** output = (float**)malloc(n_samples * sizeof(float*));
    for (i = 0; i < n_samples; i++) {
        output[i] = (float*)calloc(n_outputs, sizeof(float));
        for (j = 0; j < n_outputs; j++) {
            for (k = 0; k < discriminator->n_inputs; k++) {
                output[i][j] += generator_output[i][k] * discriminator->weights[k][j];
            }
            // Apply the activation function
            output[i][j] = sigmoid(output[i][j]);
        }
    }
    return output;
}

float** forwardPropCNNgenerator(CNN* generator, float* generator_input) {
    float** generator_output = forwardPropCNN_B(generator, generator_input);
    return generator_output;
}

float** forwardPropCNNfromGan(CNN* cnn, float** gan_data) {
    int i, j;
    float** cnn_output = (float**)malloc(sizeof(float*) * cnn->n_outputs);
    for (i = 0; i < cnn->n_outputs; i++) {
        cnn_output[i] = (float*)malloc(sizeof(float) * n_samples);
    }
    for (i = 0; i < n_samples; i++) {
        float* input = gan_data[i];
        float* output = forwardPropCNN_A(cnn, input);
        for (j = 0; j < cnn->n_outputs; j++) {
            cnn_output[j][i] = output[j];
        }
    }
    return cnn_output;
}

void forwardPropCNN_A(CNN* cnn, float* input) {
    // Perform forward propagation
    for (int i = 0; i < cnn->n_layers; i++) {
        // Multiply input by weights and add biases
        for (int j = 0; j < cnn->layer_sizes[i]; j++) {
            cnn->input[i][j] = dotProduct(cnn->weights[i][j], input, cnn->n_inputs) + cnn->biases[i][j];
        }
        // Apply activation function
        for (int j = 0; j < cnn->layer_sizes[i]; j++) {
            cnn->input[i][j] = sigmoid(cnn->input[i][j]);
        }
        // Set input for next layer
        input = cnn->input[i];
    }
}

void backpropCNN_A(CNN* cnn, float** delta, int j) {
    float** delta_input = allocate2DArray(cnn->n_inputs, cnn->n_layers);

    for (int i = cnn->n_layers - 1; i >= 0; i--) {
        if (i == cnn->n_layers - 1) {
            delta_input[i] = elementWiseMultiply(delta[i], sigmoid_derivative(cnn->input[i], cnn->layer_sizes[i]), cnn->layer_sizes[i]);
        }
        else {
            delta_input[i] = elementWiseMultiply(dot(transpose(cnn->weights[i + 1], cnn->layer_sizes[i], cnn->layer_sizes[i + 1]), delta_input[i + 1], cnn->layer_sizes[i]), sigmoid_derivative(cnn->input[i], cnn->layer_sizes[i]), cnn->layer_sizes[i]);
        }
    }

    for (int i = 0; i < cnn->n_layers; i++) {
        float** gradients = dot(delta_input[i], transpose(generatedData[j], cnn->n_inputs), cnn->n_inputs, cnn->layer_sizes[i]);
        cnn->weights[i] = subtract2DArray(cnn->weights[i], gradients, cnn->n_inputs, cnn->layer_sizes[i]);
        cnn->biases[i] = subtract1DArray(cnn->biases[i], delta_input[i], cnn->layer_sizes[i]);
    }

    deallocate2DArray(delta_input, cnn->n_layers);
}

void updateWeightsCNN(CNN* cnn, float** newWeights) {
    for (int i = 0; i < cnn->n_layers; i++) {
        for (int j = 0; j < cnn->layer_sizes[i]; j++) {
            cnn->weights[i][j] = newWeights[i][j];
        }
    }
}

void trainCNN_A(CNN* cnn, float** generatedData, float** labels) {
    int n_samples = sizeof(generatedData) / sizeof(generatedData[0]);

    for (int i = 0; i < n_samples; i++) {
        float** output = forwardPropCNN_A(cnn, generatedData[i]);
        float** error = subtract2DArray(labels[i], output, cnn->n_outputs);
        float** delta = error;
        for (int j = cnn->n_layers - 1; j >= 0; j--) {
            delta = backpropCNN_A(cnn, delta, j);                        
        }
        updateWeightsCNN(cnn, delta);
    }
}

void trainCNN_B(CNN* cnn, float** generatedData, float** labels) {
    int n_samples = sizeof(generatedData) / sizeof(generatedData[0]);
    float** output;
    float** error;
    float** delta;
    float** delta_w;
    float** delta_b;
    float** delta_input;
    float** gradients;
    float learning_rate = 0.1;

    for (int i = 0; i < n_samples; i++) {
        output = forwardPropCNN_B(cnn, generatedData[i]);
        error = subtract2DArray(labels[i], output, cnn->n_outputs);
        delta = elementWiseMultiply(error, sigmoidDerivative(output, cnn->n_outputs), cnn->n_outputs);
        delta_input = dot(transpose(cnn->weights, cnn->n_layers, cnn->layer_sizes); , delta, cnn->layer_sizes, cnn->n_layers, cnn->n_inputs);
        gradients = dot(delta, transpose(generatedData[i], cnn->n_inputs), cnn->n_outputs, cnn->n_inputs, 1);
        delta_w = scalarMultiply(learning_rate, gradients);
        delta_b = scalarMultiply(learning_rate, delta);
        cnn->weights = subtract2DArray(cnn->weights, delta_w, cnn->n_layers, cnn->layer_sizes);
        cnn->biases = subtract1DArray(cnn->biases, delta_b, cnn->n_layers);
        backprop(cnn, delta_input);
    }
}

void backpropCNN_B(CNN* cnn, float** delta_input) {
    int n_layers = cnn->n_layers;
    int* layer_sizes = cnn->layer_sizes;

    float** delta_output = allocate2DArray(layer_sizes[n_layers - 1], layer_sizes[n_layers]);

    // Backpropagation
    for (int i = n_layers - 1; i >= 0; i--) {
        int n_nodes = layer_sizes[i];
        int n_next_nodes = layer_sizes[i + 1];

        float** delta = i == n_layers - 1 ? delta_input : delta_output;
        float** weights = cnn->weights[i];
        float** biases = cnn->biases[i];

        for (int j = 0; j < n_nodes; j++) {
            float error = 0.0;
            for (int k = 0; k < n_next_nodes; k++) {
                error += delta[k][j] * weights[j][k];
            }

            delta_output[j][i] = error * sigmoid_derivative(biases[j][i]);
        }

        // Update weights and biases
        for (int j = 0; j < n_nodes; j++) {
            for (int k = 0; k < n_next_nodes; k++) {
                weights[j][k] -= LEARNING_RATE * delta_output[j][i] * biases[j][i];
            }
            biases[j][i] -= LEARNING_RATE * delta_output[j][i];
        }
    }

    deallocate2DArray(delta_output, n_layers);
}

float** calculateOutputError(CNN* generator, float generator_loss) {
    int n_outputs = generator->n_outputs;
    float** output_error = allocateMatrix(1, n_outputs);
    for (int i = 0; i < n_outputs; i++) {
        output_error[0][i] = generator_loss * generator->output[0][i];
    }
    return output_error;
}

void backpropCNNgenerator(CNN* generator, float generator_loss) {
    // Calculate the error for each output node
    float** error = calculateOutputError(generator->output, generator_loss);

    // Backpropagate the error through the hidden layers
    for (int i = generator->n_layers - 1; i >= 1; i--) {
        error = backpropLayer(generator->layers[i], error);
    }

    // Update the weights of the CNN
    updateWeights(generator, error);

    // Clean up
    freeMatrix(error);
}

float* generate1DCNNData(int n_samples, int n_inputs) {
    float* data = (float*)malloc(sizeof(float) * n_samples * n_inputs);
    int i, j;

    for (i = 0; i < n_samples; i++) {
        for (j = 0; j < n_inputs; j++) {
            data[i * n_inputs + j] = (float)rand() / (float)(RAND_MAX);
        }
    }
    return data;
}

float** generate2DCNNData(int n_samples, int n_inputs) {
    int i, j;
    float** generatedData = (float**)malloc(sizeof(float*) * n_samples);

    for (i = 0; i < n_samples; i++) {
        generatedData[i] = (float*)malloc(sizeof(float) * n_inputs);
        for (j = 0; j < n_inputs; j++) {
            // generate a random float between 0 and 1 for each input
            generatedData[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
    return generatedData;
}
