#ifndef CARL_H
#define CARL_H

#include "../CNN/CNN.h"
#include "../Discriminator/Discriminator.h"

typedef struct CA_Protocol {
    CNN* generator;
    Discriminator* discriminator;
    float** real_data;
    int batch_size;
    float generator_loss;
    float discriminator_loss;
    float** generator_output;
    float** discriminator_output;
    float** generator_error;
    float** discriminator_error;
    float generator_learning_rate;
    float discriminator_learning_rate;
    int n_inputs;
    int n_outputs;
    int n_generator_layers;
    int n_discriminator_layers;
    int* generator_layer_sizes;
    int* discriminator_layer_sizes;
    float** generator_weights;
    float** generator_biases;
    float** discriminator_weights;
    float** discriminator_biases;
    int* generator_filter_sizes;
    int generator_n_filters;
    int* discriminator_filter_sizes;
    int discriminator_n_filters;
    int generator_batch_size;
    int discriminator_batch_size;
    int generator_epochs;
    int discriminator_epochs;
    int generator_steps;
    int discriminator_steps;
    int generator_iterations;
    int discriminator_iterations;
    int generator_batch_iterations;
    int discriminator_batch_iterations;
    int generator_batch_steps;
    int discriminator_batch_steps;
} CA_Protocol;

typedef struct ARL_Protocol {
    GAN* gan;
    RL* rl;
    int batch_size;
    int n_steps;
    int current_step;
    float generator_loss;
    float discriminator_loss;
    float** generator_error;
    float** discriminator_error;
} ARL_Protocol;

CA_Protocol* initCA_Protocol(CNN* generator, Discriminator* discriminator, int batch_size, int n_inputs, int n_outputs);
ARL_Protocol* initARL_Protocol(ARL_Protocol* protocol, GAN* gan, RL* rl, int batch_size, int n_steps);
void trainCA_Protocol(CA_Protocol* protocol, float** real_data);
void trainARL_Protocol(ARL_Protocol* protocol, int n_steps);

#endif