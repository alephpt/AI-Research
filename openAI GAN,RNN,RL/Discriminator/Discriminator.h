#ifndef DISCRIMINATOR_H
#def DISCRIMINATOR_H

typedef struct Discriminator {
    //    CNN* cnn;
    int n_inputs;
    int n_outputs;
    int n_layers;
    int* layer_sizes;
    float** weights;
    float** biases;
} Discriminator;

Discriminator* createDiscriminator(int n_inputs, int n_outputs, int n_layers, int* layer_sizes);
float computeDiscriminatorLoss(discriminator_output, rl->real_data);
float computeGeneratorLoss(float* discriminator_output, int batch_size);

#endif