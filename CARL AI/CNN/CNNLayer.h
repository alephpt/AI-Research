#ifndef CNN_LAYER_H
#define CNN_LAYER_H

typedef enum {
    RELU,
    SIGMOID,
    TANH,
    SOFTMAX
} ActivationFunction;

typedef struct CNNLayer {
    int n_inputs;
    int n_outputs;
    int** neurons;
    float** weights;
    float** biases;
    float activation_derivative;
    ActivationFunction activation_function;
} CNNLayer;


CNNLayer* initializeCNNLayer(int n_inputs, int n_outputs, ActivationFunction activation_function);

#endif 
