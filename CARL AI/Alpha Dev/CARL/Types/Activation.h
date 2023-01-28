#pragma once


typedef enum Activation {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    SOFTMAX
} Activation;


float activation_derivative(Activation activation_type, float output);
float activation(Activation activation_type, float output);
