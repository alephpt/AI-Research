#include "SNN.h"
#include "../Types/Activation.h"
#include <stdlib.h>
#include <stdio.h>

float** createConnectivityMatrix(SNN* snn) {
    printf("hit testConnectivityMatrix()");
    float** connectivity_matrix = new float*[snn->n_neurons];
    printf("connectivity_matrix initialized");

    for (int i = 0; i < snn->n_neurons; i++) {
        for (int j = 0; j < snn->n_neurons; j++) {
            connectivity_matrix[i][j] = 0.0f;
        }
    }
    printf("connectiviy_matrix populated");

    for (int i = 0; i < snn->n_neurons; i++) {
        for (int j = 0; j < snn->neurons[i].n_synapses; j++) {
            connectivity_matrix[i][snn->neurons[i].synapses[j].post_neuron->index] = 1.0f;
        }
    }
    printf("connectiviy_matrix mapped");

    return connectivity_matrix;
}

int synapseLimit(int n_neurons) {
    return n_neurons * (n_neurons - 1) / 2;
}

void initSNN(SNN* snn, int n_in, int n_n, int n_syn, int n_sp, Activation activation_type, float t) {
    snn->n_inputs = n_in;
    snn->n_neurons = n_n;
    snn->neurons = new Neuron[n_n];
    snn->t = t;

    for (int n_i = 0; n_i < n_n; n_i++) {
        initNeuron(&snn->neurons[n_i], n_i, new float[n_in] , 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, t,
                   new Membrane, n_syn, new Synapse[n_syn], n_sp, new Spike[n_sp], activation_type);
    }
}