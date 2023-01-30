#pragma once
#include <map>
#include <string>

typedef enum Activation {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    SOFTPLUS,
    SOFTMAX,
    GAUSSIAN
} Activation;


const std::map<const Activation, const std::string> getActivationString = {
     {RELU, "RELU"}, 
     {LEAKY_RELU, "LEAKY RELU"}, 
     {SIGMOID, "SIGMOID"}, 
     {TANH, "TANH"}, 
     {SOFTPLUS, "SOFTPLUS"},
     {SOFTMAX, "SOFTMAX"},
     {GAUSSIAN, "GAUSSIAN"}
};

float activationDerivative(Activation activation_type, float output);
float activation(Activation activation_type, float output);
