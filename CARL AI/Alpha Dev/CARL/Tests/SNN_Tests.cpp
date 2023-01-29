#include "SNN_Tests.h"
#include "../Types/Matrix.h"
#include "../Types/Activation.h"
#include "../Types/General.h"
#include <stdio.h>

int n_inputs = 5;
int n_neurons = 4;
int n_spikes = 0;
float density = 1.0f;


void testinitSNN() {
	printf("hit testinitSNN()\n");

	SNN* snn = new SNN;
	printf("new SNN*\n");

	int n_synapses = synapseLimit(n_neurons, density);
	Activation activation_type = TANH;
	
	initSNN(snn, n_inputs, n_neurons, n_synapses, n_spikes, activation_type);

	printf("hit initSNN(snn, %d, %d, %d, %d, %s, %lf)\n", 
			n_inputs, n_neurons, n_synapses, n_spikes, getActivationString.at(activation_type).c_str(), 0.0f);
	
	printSNN(snn);
	return;
}

void testConnectivityMatrix() {
	printf("hit testConnectivityMatrix()\n");
	SNN* snn = new SNN;
	printf("new SNN*\n");

	int n_synapses = synapseLimit(n_neurons, density);
	Activation activation_type = TANH;
	
	initSNN(snn, n_inputs, n_neurons, n_synapses, n_spikes, activation_type);
	printf("hit initSNN(snn, %d, %d, %d, %d, %s, %lf)\n", 
			n_inputs, n_neurons, n_synapses, n_spikes, getActivationString.at(activation_type).c_str(), 0.0f);

	float** conn_matrix = createConnectivityMatrix(snn);

	printMatrix(conn_matrix, snn->n_neurons, snn->n_neurons);
	return;
}