#include "GAN.h"

GAN* createGAN(int n_inputs, int n_outputs, int n_layers, int* filter_sizes, int n_filters, int* layer_sizes) {
    GAN* gan = (GAN*)malloc(sizeof(GAN));
    gan->n_inputs = n_inputs;
    gan->n_outputs = n_outputs;
    gan->weights = allocate2DArray(n_inputs, n_outputs);
    gan->generator = createCNN(n_inputs, n_outputs, n_layers, filter_sizes, n_filters, layer_sizes);
    gan->discriminator = createDiscriminator(n_inputs, n_outputs, n_layers, layer_sizes);
    return gan;
}

void destroyGAN(GAN* gan) {
    destroyCNN(gan->generator);
    destroyCNN(gan->discriminator);
    free(gan);
}

float** forwardPropGANfromCNN(GAN* gan, float** cnn_data) {
    float** generator_input = cnn_data;
    float** generator_output = forwardPropCNNgenerator(gan->generator, generator_input);
    float** discriminator_output = forwardPropCNNdiscriminator(gan->discriminator, generator_output);
    return discriminator_output;
}

void backpropGAN(GAN* gan, float generator_loss, float discriminator_loss) {
    backpropCNN(gan->generator, generator_loss);
    backpropDiscriminator(gan->discriminator, discriminator_loss);
}

float** generate2DGANData(GAN* gan, int n_samples) {
    float** generator_input = generate1DDataCNN(n_samples, gan->n_inputs);
    float** generated_data = forwardPropCNN(gan->generator, generator_input);

    return generatedData;
}

void updateWeightsGAN(GAN* gan, float** newWeights) {
    for (int i = 0; i < gan->n_layers; i++) {
        for (int j = 0; j < gan->layer_sizes[i]; j++) {
            for (int k = 0; k < gan->layer_sizes[i + 1]; k++) {
                gan->weights[i][j][k] = newWeights[i][j][k];
            }
        }
    }
}

void trainGAN(RL* rl) {
    // Retrieve generator and discriminator from GAN
    CNN* generator = rl->gan->generator;
    Discriminator* discriminator = rl->gan->discriminator;

    // Forward propagate generator with random noise
    float** generator_input = generateRandomNoise(rl->gan->n_inputs, rl->batch_size);
    float** generator_output = forwardPropCNN_A(generator, generator_input);

    // Forward propagate discriminator with generator output
    float** discriminator_output = forwardPropCNN_A(discriminator, generator_output);

    // Compute loss and backpropagate
    float generator_loss = computeGeneratorLoss(discriminator_output);
    float discriminator_loss = computeDiscriminatorLoss(discriminator_output, rl->real_data);
    backpropGAN(rl->gan, generator_loss, discriminator_loss);

    // Update generator and discriminator weights
    updateWeightsCNN(generator);
    updateWeightsCNN(discriminator);
}