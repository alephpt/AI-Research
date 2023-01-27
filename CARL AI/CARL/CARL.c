#include "../utilities/shop.h"

CA_Protocol* initCA_Protocol(CNN* generator, Discriminator* discriminator, int batch_size, int n_inputs, int n_outputs) {
    CA_Protocol* protocol = (CA_Protocol*)malloc(sizeof(CA_Protocol));
    protocol->generator = generator;
    protocol->discriminator = discriminator;
    protocol->batch_size = batch_size;
    protocol->n_inputs = n_inputs;
    protocol->n_outputs = n_outputs;
    protocol->n_generator_layers = generator->n_layers;
    protocol->n_discriminator_layers = discriminator->layers;
    protocol->generator_layer_sizes = generator->layer_sizes;
    protocol->discriminator_layer_sizes = discriminator->layer_sizes;
    protocol->generator_filter_sizes = generator->filter_sizes;
    protocol->generator_n_filters = generator->n_filters;
    protocol->discriminator_filter_sizes = discriminator->filter_sizes;
    protocol->discriminator_n_filters = discriminator->n_filters;
    protocol->generator_batch_size = generator->batch_size;
    protocol->discriminator_batch_size = discriminator->batch_size;
    protocol->generator_epochs = generator->epochs;
    protocol->discriminator_epochs = discriminator->epochs;
    protocol->generator_steps = generator->steps;
    protocol->discriminator_steps = discriminator->steps;
    protocol->generator_iterations = 0;
    protocol->discriminator_iterations = 0;
    protocol->generator_batch_iterations = 0;
    protocol->discriminator_batch_iterations = 0;
    protocol->generator_batch_steps = 0;
    protocol->discriminator_batch_steps = 0;
    protocol->generator_learning_rate = generator->learning_rate;
    protocol->discriminator_learning_rate = discriminator->learning_rate;
    protocol->real_data = allocateMatrix(protocol->batch_size, protocol->n_inputs);
    protocol->generator_output = allocateMatrix(protocol->batch_size, protocol->n_outputs);
    protocol->discriminator_output = allocateMatrix(protocol->batch_size, protocol->n_outputs);
    protocol->generator_error = allocateMatrix(protocol->batch_size, protocol->n_outputs);
    protocol->discriminator_error = allocateMatrix(protocol->batch_size, protocol->n_outputs);
    protocol->discriminator_error = allocateMatrix(protocol->batch_size, protocol->n_outputs);
    protocol->generator_learning_rate = 0.01;
    protocol->discriminator_learning_rate = 0.01;
    protocol->n_inputs = n_inputs;
    protocol->n_outputs = n_outputs;
    protocol->n_generator_layers = generator->n_layers;
    protocol->n_discriminator_layers = discriminator->n_layers;
    protocol->generator_layer_sizes = (int*)malloc(sizeof(int) * generator->n_layers);
    protocol->discriminator_layer_sizes = (int*)malloc(sizeof(int) * discriminator->n_layers);
    protocol->generator_weights = allocateMatrix(generator->n_layers, n_inputs);
    protocol->generator_biases = allocateMatrix(generator->n_layers, 1);
    protocol->discriminator_weights = allocateMatrix(discriminator->n_layers, n_inputs);
    protocol->discriminator_biases = allocateMatrix(discriminator->n_layers, 1);
    protocol->generator_filter_sizes = (int*)malloc(sizeof(int) * n_generator_layers);
    protocol->generator_n_filters = generator->n_filters;
    protocol->discriminator_filter_sizes = (int*)malloc(sizeof(int) * discriminator->n_layers);
    protocol->discriminator_n_filters = discriminator->n_filters;
    protocol->generator_batch_size = generator->batch_size;
    protocol->discriminator_batch_size = discriminator->batch_size;
    protocol->generator_epochs = generator->epochs;
    protocol->discriminator_epochs = discriminator->epochs;
    protocol->generator_steps = generator->steps;
    protocol->discriminator_steps = discriminator->steps;
    protocol->generator_iterations = generator->iterations;
    protocol->discriminator_iterations = discriminator->iterations;
    protocol->generator_batch_iterations = generator->batch_iterations;
    protocol->discriminator_batch_iterations = discriminator->batch_iterations;
    protocol->generator_batch_steps = generator->batch_steps;
    protocol->discriminator_batch_steps = discriminator->batch_steps;
    return protocol;
}


void initARL_Protocol(ARL_Protocol* protocol, GAN* gan, RL* rl, int batch_size, int n_steps) {
    ARL_Protocol* protocol = (ARL_Protocol*)malloc(sizeof(ARL_Protocol));
    protocol->gan = gan;
    protocol->rl = rl;
    protocol->batch_size = batch_size;
    protocol->n_steps = n_steps;
    protocol->current_step = 0;
    protocol->generator_loss = 0;
    protocol->discriminator_loss = 0;
    protocol->generator_error = allocateMatrix(batch_size, gan->generator->n_outputs);
    protocol->discriminator_error = allocateMatrix(batch_size, gan->discriminator->n_outputs);
}


void trainCA_Protocol(CA_Protocol* protocol, float** real_data) {
    protocol->real_data = real_data;

    // training generator
    float** generator_input = randomNoise(protocol->batch_size, protocol->n_inputs);
    float** generator_output = forwardPropCNN(protocol->generator, generator_input);
    float** discriminator_output = forwardPropCNN(protocol->discriminator, generator_output);
    protocol->generator_loss = generatorLoss(discriminator_output);
    protocol->generator_error = generatorError(discriminator_output);
    backPropCNN(protocol->generator, protocol->generator_error, protocol->generator_learning_rate);

    // training discriminator
    float** real_output = forwardPropCNN(protocol->discriminator, real_data);
    float** fake_output = forwardPropCNN(protocol->discriminator, generator_output);
    protocol->discriminator_loss = discriminatorLoss(real_output, fake_output);
    protocol->discriminator_error = discriminatorError(real_output, fake_output);
    backPropCNN(protocol->discriminator, protocol->discriminator_error, protocol->discriminator_learning_rate);
}

void trainARL_Protocol(ARL_Protocol* protocol, int n_steps) {
    for (int i = 0; i < n_steps; i++) {
        // Train GAN
        trainGAN(protocol->gan, protocol->batch_size);
        // Update generator weights and biases in RL
        updateRL(protocol->rl, protocol->gan->generator_weights, protocol->gan->generator_biases);
        // Train RL
        trainRL(protocol->rl, protocol->batch_size);
        // Update discriminator weights and biases in GAN
        updateGAN(protocol->gan, protocol->rl->discriminator_weights, protocol->rl->discriminator_biases);
    }
}

