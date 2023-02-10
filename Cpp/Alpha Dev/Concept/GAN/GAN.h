#ifndef GAN_H
#define GAN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../CNN/CNN.h"
#include "../Discriminator/Discriminator.h"
#include "../utilities/shop.h"

// need to add generator bias'
typedef struct GAN {
    CNN* generator;
    Discriminator* discriminator;
    int n_inputs;
    int n_outputs;
    float** weights;
} GAN;

GAN* createGAN(int n_inputs, int n_outputs, int n_layers, int* filter_sizes, int n_filters, int* layer_sizes);
float** generate2DGANData(GAN* gan, int n_samples);
void destroyGAN(GAN* gan);
void trainGAN(GAN* gan, int batch_size);
void updateWeightsGAN(GAN* gan, float** newWeights);
float** forwardPropGANfromCNN(GAN* gan, float** cnn_data);
void backpropGAN(GAN* gan, float generator_loss, float discriminator_loss);
void saveGAN(GAN* gan, char* filename);
void loadGAN(GAN* gan, char* filename);
void receiveDataFromCNNandUpdate(GAN* gan, char* buffer);
void sendDataToCNN(GAN* gan, char* buffer);

#endif
