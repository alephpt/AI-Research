#include "SNN_Tests.h"
#include "../Types/Matrix.h"
#include "../Types/Activation.h"
#include <stdio.h>

int n_inputs = 10;
int n_neurons = 10;
int n_spikes = 3;

void testinitSNN() {
	printf("hit testinitSNN()\n");
	SNN* snn = new SNN;
	printf("new SNN*\n");
	int n_synapses = synapseLimit(n_neurons);
	initSNN(snn, n_inputs, n_neurons, n_synapses, n_spikes, TANH, 0.0f);
	printf("hit initSNN(snn, %d, %d, %d, %d, TANH, %f)\n", n_inputs, n_neurons, n_synapses, n_spikes, 0.0f);
	return;
}

void testConnectivityMatrix() {
	SNN* snn = new SNN;


	initSNN(snn, n_inputs, n_neurons, synapseLimit(n_neurons), n_spikes, RELU, 0.0f);

	float** conn_matrix = createConnectivityMatrix(snn);

	printMatrix(conn_matrix, snn->n_neurons, snn->n_neurons);
	return;
}