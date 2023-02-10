#pragma once
#include "Scalar.h"
#include <string>

const std::string activationString[] = { 
    "RELU", 
    "LEAKY RELU", 
    "SIGMOID", 
    "TANH", 
    "SOFTPLUS", 
    "SOFTMAX", 
    "GAUSSIAN", 
    "RELU_DERIVATIVE",
    "LEAKY RELU_DERIVATIVE", 
    "SIGMOID_DERIVATIVE", 
    "TANH_DERIVATIVE", 
    "SOFTPLUS_DERIVATIVE", 
    "SOFTMAX_DERIVATIVE", 
    "GAUSSIAN_DERIVATIVE"

};

typedef enum Activation {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    SOFTPLUS,
    SOFTMAX,
    GAUSSIAN,
    RELU_DERIVATIVE,
    LEAKY_RELU_DERIVATIVE,
    SIGMOID_DERIVATIVE,
    TANH_DERIVATIVE,
    SOFTPLUS_DERIVATIVE,
    SOFTMAX_DERIVATIVE,
    GAUSSIAN_DERIVATIVE
} Activation;



fscalar activation(Activation activation_type, fscalar output);
