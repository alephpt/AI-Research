#include "SNN.h"
#include "../Types/General.h"
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

int synapseLimit(int n_neurons, float density) {
    return (int)((float)(n_neurons * (n_neurons - 1) / 2) * density);
}

void initSNN(SNN* snn, int n_in, int n_n, int n_syn, int n_sp, Activation activation_type) {
    snn->n_inputs = n_in;
    snn->n_neurons = n_n;
    snn->n_synapses = n_syn;
    snn->n_spikes = n_sp;
    snn->neurons = new Neuron[n_n];
    snn->t = getTime();

    for (int n_i = 0; n_i < n_n; n_i++) {
        initNeuron(&snn->neurons[n_i], n_i, new float[n_n - 1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   new Membrane, (n_n - 1), new Synapse[n_n - 1], n_sp, new Spike[n_sp], activation_type);
    }
}

Neuron* getNeuron(SNN* snn, int idx) {
    return &snn->neurons[idx];
}

void printSNN(SNN* snn) {
    printf("\t~~ / SNN \\ ~~\n");
    printf("snn # of inputs: %d\n", snn->n_inputs);
    printf("snn # of neurons: %d\n", snn->n_neurons);
    printf("snn # of synapses: %d\n", snn->n_synapses);
    printf("snn # of spikes: %d\n\n", snn->n_spikes);
    printf("Neurons:\n");
    
    for (int n_i = 0; n_i < snn->n_neurons; n_i++) {
        printNeuron(&snn->neurons[n_i], n_i, snn->n_neurons - 1);
    }
}