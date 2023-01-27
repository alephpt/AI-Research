#ifndef DISCRIMINATOR_H
#def DISCRIMINATOR_H

#include "../CNN/CNN.h"
#include "../CNN/CNNLayer.h"
#include "../utilities/shop.h"

typedef struct Discriminator {
    float** input;
    float** output;
    float learning_rate;
    int n_inputs;
    int n_outputs;
    float** weights;
    float** biases;
    CNNLayer* layers;
    int* layer_sizes;
    int n_layers;
    int batch_size;
} Discriminator;

Discriminator* createDiscriminator(int n_inputs, int n_outputs, int n_layers, int* layer_sizes);
float computeDiscriminatorLoss(float** discriminator_output, float** real_data, int batch_size);
float computeGeneratorLoss(float* discriminator_output, int batch_size);
void backpropDiscriminator(Discriminator* discriminator, float discriminator_loss);
void updateDiscriminatorWeights(float** weights, float** output, float** error, float learning_rate);
void syncDiscriminatorToCNN(CNN* generator, Discriminator* discriminator);

#endif