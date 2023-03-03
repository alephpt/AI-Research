#include "SNN_Tests.h"
#include "../Types/Matrix.h"
#include "../Types/Activation.h"
#include <stdio.h>

int n_inputs = 3;
int n_neurons = 100000;
int n_spikes = 2;
fscalar density = 1.0f;


void testinitSNN() {
	Activation activation_type = TANH;
	int n_synapses = synapseLimit(n_neurons, density);
	SNN* snn = new SNN;
	printf("Initialized new SNN*\n");

	initSNN(snn, n_inputs, n_neurons, n_synapses, n_spikes, activation_type);

	printf("hit initSNN(snn, %d, %d, %d, %d, %s, %lf)\n\n",
		n_inputs, n_neurons, n_synapses, n_spikes, activationString[activation_type].c_str(), 0.0f);

	//printSNN(snn);
	printf("Finished!");

	return;
}

void testConnectivityMatrix() {
	printf("hit testConnectivityMatrix()\n");
	SNN* snn = new SNN;
	printf("new SNN*\n");

	int n_synapses = synapseLimit(n_neurons, density);
	Activation activation_type = LEAKY_RELU;
	
	printf("hit initSNN(snn, %d, %d, %d, %d, %s, %lf)\n\n", 
			n_inputs, n_neurons, n_synapses, n_spikes, activationString[activation_type].c_str(), 0.0f);
	initSNN(snn, n_inputs, n_neurons, n_synapses, n_spikes, activation_type);


	fscalar** conn_matrix = createConnectivityMatrix(snn);

	//printMatrix(conn_matrix, snn->n_neurons, snn->n_neurons);
	return;
}