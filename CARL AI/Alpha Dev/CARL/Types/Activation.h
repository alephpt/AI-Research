#pragma once
#include <map>
#include <string>

typedef enum Activation {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    SOFTMAX
} Activation;


const std::map<const Activation, const std::string> getActivationString = {
     {RELU, "RELU"}, 
     {LEAKY_RELU, "LEAKY RELU"}, 
     {SIGMOID, "SIGMOID"}, 
     {TANH, "TANH"}, 
     {SOFTMAX, "SOFTMAX"} 
};

float activationDerivative(Activation activation_type, float output);
float activation(Activation activation_type, float output);
