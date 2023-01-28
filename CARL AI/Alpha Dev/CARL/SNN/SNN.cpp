#include "SNN.h"
#include <stdlib.h>

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

void initSNN(SNN* snn, int n_inputs, int n_neurons, Activation activation_type, float t) {
    snn->n_inputs = n_inputs;
    snn->n_neurons = n_neurons;
    snn->neurons = new Neuron[n_neurons];
    snn->t = t;

    // Initialize Neurons
    for (int i = 0; i < n_neurons; i++) {
        Neuron* neuron = &snn->neurons[i];
        neuron->index = i;
        neuron->weights = new float[n_inputs + 1]; // n_inputs + 1 for bias
        neuron->bias = 0.0f;
        neuron->input = 0.0f;
        neuron->output = 0.0f;
        neuron->delta = 0.0f;
        neuron->memberane_potential = 0.0f;
        neuron->membrane.capacitance = 0.0f;
        neuron->membrane.resistance = 0.0f;
        neuron->membrane.V_rest = 0.0f;
        neuron->membrane.V_threshold = 0.0f;
        neuron->membrane.t = 0.0f;
        neuron->n_synapses = 0;
        neuron->synapses = createNewSynapse(new Synapse, 0.0f, 0.0f, NULL, NULL);
        neuron->n_spikes = 0;
        neuron->spikes = createNewSpike(new Spike);
        neuron->activation_type = activation_type;
    }
}