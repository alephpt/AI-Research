#include "Activation.h"
#include <math.h>
#include <stdio.h>

static const fscalar pi = 3.1415926f;

    // SIGMOIDS //
static inline fscalar sigmoid(fscalar x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline fscalar sigmoid_derivative(fscalar x) {
    fscalar s = sigmoid(x);
    return s * (1.0f - s);
}


    // TANH //
static inline fscalar tanh_activation(fscalar x) {
    return (expf(x) - expf(-x)) / (expf(x) + expf(-x));
}

static inline fscalar tanh_derivative(fscalar x) {
    return 1.0f - powf(tanh_activation(x), 2.0f);
}


    // RELU //
static inline fscalar relu(fscalar x) {
    if (x < 0) { return 0.0f; }
    return x;
}

static inline fscalar relu_derivative(fscalar x) {
    if (x < 0) { return 0.0f; }
    return 1.0f;
}


    // LEAKY_RELU //
static inline fscalar leaky_relu(fscalar x, fscalar alpha) {
    if (x > 0) { return x; }
    else { return alpha * x; }
}

static inline fscalar leaky_relu_derivative(fscalar x, fscalar alpha) {
    if (x > 0) { return 1.0f; }
    else { return alpha; }
}

    // SOFTPLUS //
static inline fscalar softplus(fscalar x) {
    return logf(1.0f + expf(x));
}

static inline fscalar softplus_derivative(fscalar x) {
    return 1.0f / (1.0f + expf(-x));
}

    // SOFTMAX //
static inline fscalar softmax(fscalar x) {
    return expf(x) / (expf(x) + 1.0f);
}

static inline fscalar softmax_derivative(fscalar x) {
    fscalar s = softmax(x);
    return s * (1.0f - s);
}

    // GAUSSIAN //
static inline fscalar gaussian(fscalar x) {
    fscalar mean = 0.0f;
    fscalar standard_deviation = 1.0f;

    return (1.0f / (standard_deviation * sqrtf(2.0f * pi))) * expf(-0.5f * powf((x - mean) / standard_deviation, 2.0f));
}

static inline fscalar gaussian_derivative(fscalar x) {
    return -x * expf(-x * x / 2.0f) / sqrtf(2.0f * pi);
}

// ACTIVATION FUNCTIONS //
// TODO: *Activation[]
fscalar activation(Activation activation_type, fscalar output) {
    if (activation_type == SIGMOID_DERIVATIVE) {
        return sigmoid_derivative(output);
    }
    else if (activation_type == TANH_DERIVATIVE) {
        return tanh_derivative(output);;
    }
    else if (activation_type == RELU_DERIVATIVE) {
        return relu_derivative(output);
    }
    else if (activation_type == LEAKY_RELU_DERIVATIVE) {
        return leaky_relu_derivative(output, 0.01f);
    }
    else if (activation_type == SOFTPLUS_DERIVATIVE) {
        return softplus_derivative(output);
    }
    else if (activation_type == SOFTMAX_DERIVATIVE) {
        return softmax_derivative(output);
    }
    else if (activation_type == GAUSSIAN_DERIVATIVE) {
        return gaussian_derivative(output);
    } else
    if (activation_type == SIGMOID) {
        return sigmoid(output);
    }
    else if (activation_type == TANH) {
        return tanh_activation(output);;
    }
    else if (activation_type == RELU) {
        return relu(output);
    }
    else if (activation_type == LEAKY_RELU) {
        return leaky_relu(output, 0.01f);
    }
    else if (activation_type == SOFTPLUS) {
        return softplus(output);
    }
    else if (activation_type == SOFTMAX) {
        return softmax(output);
    }
    else if (activation_type == GAUSSIAN) {
        return gaussian(output);
    }
    else {
        printf("ACTIVATION ERROR: Invalid activation type\n");
        return 0;
    }
}

