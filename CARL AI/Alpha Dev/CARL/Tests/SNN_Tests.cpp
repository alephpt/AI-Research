#include "SNN_Tests.h"
#include "../Types/Matrix.h"
#include "../Types/Activation.h"

void testinitSNN() {
	return;
}

void testConnectivityMatrix() {
	SNN* snn = new SNN;
	initSNN(snn, 10, 100, RELU, 0.0f);
	float** conn_matrix = createConnectivityMatrix(snn);
	printMatrix(conn_matrix, snn->n_neurons, snn->n_neurons);
	return;
}