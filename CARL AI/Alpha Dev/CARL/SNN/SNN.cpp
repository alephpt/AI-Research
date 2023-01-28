#include "SNN.h"

float** createConnectivityMatrix(SNN* snn) {
    float** connectivity_matrix = new float*[snn->n_neurons];

    for (int i = 0; i < snn->n_neurons; i++) {
        for (int j = 0; j < snn->n_neurons; j++) {
            connectivity_matrix[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < snn->n_neurons; i++) {
        for (int j = 0; j < snn->neurons[i].n_synapses; j++) {
            connectivity_matrix[i][snn->neurons[i].synapses[j].post_neuron->index] = 1.0f;
        }
    }

    return connectivity_matrix;
}