#pragma once
#include "../Types/Types.h"

// need to implement an variational auto-encoder 
// use the adversarial loss of GANs in addition to the reconstruction loss of Autoencoders
// look at implementing an adaptive instance normalization layer
// look at implementing a spacially adaptive normalization layer
// look at implementing multiple class vectors to represent different styles

class Generator {
public:
	Generator(int n_inputs, int n_hidden, int n_outputs) {
		n_inputs = n_inputs;
		n_hidden = n_hidden;
		n_outputs = n_outputs;
		weights = new fscalar[((n_inputs + 1) * n_hidden + (n_hidden + 1) * n_outputs)];
		biases = new fscalar[(n_hidden * n_outputs)];
	}

	~Generator() {
		delete[] weights;
		delete[] biases;
	}


//private:
	int n_inputs;
	int n_hidden;
	int n_outputs;
	fscalar* weights;
	fscalar* biases;
};