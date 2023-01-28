#include "SNN_Tests.h"
#include "../Types/Matrix.h"

void testinitSNN() {
	return;
}

void testConnectivityMatrix() {
	SNN* snn = new SNN;
	float** conn_matrix = createConnectivityMatrix(snn);
	printMatrix(conn_matrix);
	return;
}