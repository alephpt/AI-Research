#include "Type_Tests.h"
#include "../Types/Types.h"
#include <stdio.h>
#include <stdlib.h>


void testGradientDescent() {
	fscalar x[] = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
	fscalar y[] = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };

	vector<fscalar> error;
	fscalar deviation;
	fscalar b0 = 0.f;
	fscalar b1 = 0.f;
	fscalar learning_rate = 0.01f;

	printf("hit tgd.\n");

	for (int i = 0; i < 5000; i++) {
		int index = i % 10;
		fscalar p = b0 + b1 * x[index];					// calculating prediction
		deviation = p - y[index];						// calculating the error
		b0 = b0 - learning_rate * deviation;
		b1 = b1 - learning_rate * deviation * x[index];
		
		if (i % 500 == 0) {
			printf("Iteration %d:\nB0: \t%f\nB1: \t%f\nError: \t%f\n\n", i, b0, b1, deviation);
		}
	}
}


void testRandomNumbers() {
	size_t width = 10;
	size_t height = 17;

	fmatrix random_values = generate2dNoise(height, width);
	printFMatrix(random_values);
}

void testActivationType() {
	fscalar x = -1.0;

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


