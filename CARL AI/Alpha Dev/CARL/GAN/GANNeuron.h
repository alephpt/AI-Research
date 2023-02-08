#include "../Types/Types.h"

typedef struct GANNeuron {
    int n_inputs;
    fscalar* weights;
    fscalar bias;
    fscalar output;
} Neuron;