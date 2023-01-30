#pragma once

// need to implement an variational auto-encoder 
// use the adversarial loss of GANs in addition to the reconstruction loss of Autoencoders
// look at implementing an adaptive instance normalization layer
// look at implementing a spacially adaptive normalization layer
// look at implementing multiple class vectors to represent different styles

class Generator {
public:
	Generator(int n_inputs, int n_hidden, int n_outputs) {
		this->n_inputs = n_inputs;
		this->n_hidden = n_hidden;
		this->n_outputs = n_outputs;
		this->weights = new float[(n_inputs + 1) * n_hidden + (n_hidden + 1) * n_outputs];
		this->biases = new float[n_hidden * n_outputs];
	}

	~Generator() {
		delete[] this->weights;
		delete[] this->biases;
	}


//private:
	int n_inputs;
	int n_hidden;
	int n_outputs;
	float* weights;
	float* biases;


};