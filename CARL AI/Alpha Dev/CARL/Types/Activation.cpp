#include "Activation.h"
#include <Math.h>
#include <stdio.h>

// RELU //
static inline float relu(float x) {
    if (x < 0) { return 0.0f; }
    return x;
}

static inline float relu_derivative(float x) {
    if (x < 0) { return 0.0f; }
    return 1.0f;
}


// LEAKY_RELU //

static inline float leaky_relu(float x, float alpha) {
    if (x > 0) { return x; }
    else { return alpha * x; }
}

static inline float leaky_relu_derivative(float x, float alpha) {
    if (x > 0) { return 1.0f; }
    else { return alpha; }
}

// TANH //

static inline float tanh(float x) {
    return (expf(x) - expf(-x)) / (expf(x) + expf(-x));
}

static inline float tanh_derivative(float x) {
    return 1.0f - powf(tanh(x), 2.0f);
}

// SIGMOIDS //
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}


// SOFTMAX //

static inline float softmax(float x) {
    return (float)(expf(x) / (expf(x) + 1));
}

static inline float softmax_derivative(float x) {
    float s = softmax(x);
    return s * (1.0f - s);
}


// ACTIVATION FUNCTIONS //

float activationDerivative(Activation activation_type, float output) {
    if (activation_type == SIGMOID) {
        return sigmoid_derivative(output);
    }

    else if (activation_type == TANH) {
        return tanh_derivative(output);;
    }
    else if (activation_type == RELU) {
        return relu_derivative(output);
    }
    else if (activation_type == LEAKY_RELU) {
        return leaky_relu_derivative(output, (float)0.01);
    }
    else if (activation_type == SOFTMAX) {
        return softmax_derivative(output);
    }
    else {
        printf("ACTIVATION DERIVATIVE ERROR: Invalid activation type\n");
        return 0;
    }
}

float activation(Activation activation_type, float output) {
    if (activation_type == SIGMOID) {
        return sigmoid(output);
    }
    else if (activation_type == TANH) {
        return tanh(output);;
    }
    else if (activation_type == RELU) {
        return relu(output);
    }
    else if (activation_type == LEAKY_RELU) {
        return leaky_relu(output, (float)0.01);
    }
    else if (activation_type == SOFTMAX) {
        return softmax(output);
    }
    else {
        printf("ACTIVATION ERROR: Invalid activation type\n");
        return 0;
    }
}

