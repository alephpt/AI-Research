#include "Type_Tests.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

void testRandomNumbers() {
	size_t width = 10;
	size_t height = 17;

	fmatrix random_values = generate2dNoise(height, width);
	printFMatrix(random_values);
}

void testActivationType() {
	float x = -1.0;

	printf("Ativation type:\n");
	printf("sigmoid(%f) = %f\n", x, activation(SIGMOID, x));
	printf("tanh(%f) = %f\n", x, activation(TANH, x));
	printf("leaky_relu(%f) = %f\n", x, activation(LEAKY_RELU, x));
	printf("relu(%f) = %f\n", x, activation(RELU, x));
	printf("softplus(%f) = %f\n", x, activation(SOFTPLUS, x));
	printf("softmax(%f) = %f\n", x, activation(SOFTMAX, x));
	printf("gaussian(%f) = %f\n", x, activation(GAUSSIAN, x));
	printf("sigmoid(%f) = %f\n", x, activation(SIGMOID_DERIVATIVE, x));
	printf("tanh(%f) = %f\n", x, activation(TANH_DERIVATIVE, x));
	printf("leaky_relu(%f) = %f\n", x, activation(LEAKY_RELU_DERIVATIVE, x));
	printf("relu(%f) = %f\n", x, activation(RELU_DERIVATIVE, x));
	printf("softplus(%f) = %f\n", x, activation(SOFTPLUS_DERIVATIVE, x));
	printf("softmax(%f) = %f\n", x, activation(SOFTMAX_DERIVATIVE, x));
	printf("gaussian(%f) = %f\n", x, activation(GAUSSIAN_DERIVATIVE, x));
	return;
}


