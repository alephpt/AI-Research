#pragma once
#include "Neuron.h"

typedef struct SNN SNN;

struct SNN {
	int n_inputs;
	int n_neurons;
	Neuron* neurons;
	float t;
};

// let's start with 10 neurons and 90 synapses
void initSNN(SNN* snn, int n_in, int n_n, int n_syn, int n_sp, Activation activation_type, float t);
float** createConnectivityMatrix(SNN* snn);
int synapseLimit(int n_neurons);

// TODO: 
/*
	- Spike Generation
	- Update Membrane Potential
	- Synapse Weight Update
	- Neuron Activity/Firing Rate
	- Network/Layer Connectivity Function
		X Define a Connectivity Matrix
		- Test the Output based on the Input
	- Input/Output Testing
	- Performance Evaluation
	- Training
	- Propagation
	- Saving and Loading Network
	- incorporate the CARL_Protocol into the spiking behaviour of the neurons
	- ADD Temporal component to CARL implementing time-based learning
	- define Time Based Learning Rules: Spike-Timing Dependent Plasticity
	- Implement Spike Rate Synchronization methods and incorporate it into CARL's evaluation process
	- Update Carls backpropagation algorithm to take into account spiking of neurons
	- Implement controls of the Input Spikes and Output Spikes
	- Incorporate Lateral Inhibition into CARL
*/