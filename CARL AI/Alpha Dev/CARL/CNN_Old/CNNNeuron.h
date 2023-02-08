#include "../Types/Types.h"

typedef struct CNNNeuron {
    int n_inputs;
    fscalar* weights;
    fscalar bias;
    fscalar output;
} CNNNeuron;