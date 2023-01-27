#ifndef CNN_H
#define CNN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "CNNLayer.h"
#include "../Discriminator/Discriminator.h"

typedef struct CNN {
    float** input;
    float** output;
    int n_inputs;
    int n_outputs;
    CNNLayer* layers;
    int n_layers;
    int* filter_sizes;
    int n_filters;
    float** weights;
    float** biases;
    int* layer_sizes;
} CNN;

CNN* createCNN(int n_inputs, int n_outputs, int n_layers, int* filter_sizes, int n_filters, int* layer_sizes);      // defined
void destroyCNN(CNN* cnn);
int predictCNN(CNN* cnn, float* testData);                                                                          // defined
float evaluateCNN(CNN* cnn, float** testData, int* testLabels);                                                     // defined
float evaluateCNNwithSS(CNN* cnn, float** testData, float* testLabels, int sample_size);                            // defined
float** forwardPropCNNdiscriminator(Discriminator* discriminator, float** generator_output);                        // defined
float** forwardPropCNNgenerator(CNN* generator, float* generator_input);                                            // defined
float** forwardPropCNNfromGAN(CNN* cnn, float** gan_data);                                                          // defined
float** forwardPropCNN_A(Discriminator* discriminator, float** generator_output);                                   // defined
void forwardPropCNN_VOID(CNN* cnn, float* input);                                                                      // defined
void forwardPropCNN_B(CNN* cnn, float* testData);                                                                   // defined
void trainCNN_A(CNN* cnn, float** generatedData, float** labels);                                                   // defined
void backpropCNN_A(CNN* cnn, float** delta, int j);                                                                 // defined
void trainCNN_B(CNN* cnn, float** generatedData, float** labels);                                                   // defined
void backpropCNN_B(CNN* cnn, float** delta_input);                                                                  // defined
float* generate1DCNNData(int n_samples, int n_inputs);                                                              // defined
float** generate2DCNNData(int n_samples, int n_inputs);                                                             // defined
void updateWeightsCNN(CNN* cnn, float** newWeights);                                                                // defined
void saveCNN(CNN* cnn, char* filename);
void loadCNN(CNN* cnn, char* filename);
void receiveDataFromGANandUpdate(CNN* cnn, char* buffer);
void sendDataToGAN(CNN* cnn, char* buffer);
void sendDataToRL(CNN* cnn, char* buffer);

#endif
