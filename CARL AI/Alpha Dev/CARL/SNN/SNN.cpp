#include "SNN.h"
#include "../Types/General.h"
#include <stdlib.h>
#include <stdio.h>

float** createConnectivityMatrix(SNN* snn) {
    printf("hit createConnectivityMatrix()");
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
            connectivity_matrix[i][snn->neurons[i].synapses[j]->post_neuron->index] = 1.0f;
        }
    }
    printf("connectiviy_matrix mapped");

    return connectivity_matrix;
}

int synapseLimit(int n_neurons, float density) {
    return (int)((float)(n_neurons * (n_neurons - 1) / 2) * density);
}

void initSNN(SNN* snn, int n_in, int nn, int n_syn, int n_sp, Activation activation_type) {
    snn->n_inputs = n_in;
    snn->n_neurons = 0;
    snn->n_synapses = n_syn;
    snn->n_spikes = n_sp;
    snn->neurons = new Neuron[nn];
    snn->t = getTime();

    printf(" - Initializing %d Neurons .. \n", nn, (((nn - 1) * nn) / 2));
    for (int ni = 0; ni < nn; ni++) {
        snn->neurons[ni].index = ni;
        snn->neurons[ni].synapses = new Synapse*[nn - 1];
        initNeuron(activation_type, &snn->neurons[ni], new Membrane, new Spike[n_sp], n_sp,
                new float[nn - 1], 0.0f, nn - 1, new float[nn - 1], nn - 1, new float[nn - 1], 0.0f, 0.0f);
        snn->n_neurons++;
    }

    int si = 0;
    printf(" - Linking %d Neurons to %d Synapses .. \n", nn, (((nn - 1) * nn) / 2));
    for (int ni = 0; ni < nn - 1; ni++) {
        for (int nt = ni + 1; nt < nn; nt++) {
            Neuron* i_neuron = &snn->neurons[ni];
            Neuron* t_neuron = &snn->neurons[nt];
            i_neuron->synapses[i_neuron->n_synapses] = createNewSynapse(new Synapse, si, 0.0f, 0.0f, i_neuron, t_neuron);
            t_neuron->synapses[t_neuron->n_synapses++] = i_neuron->synapses[i_neuron->n_synapses++];
            si++;
        }
    }
}

Neuron* getNeuron(SNN* snn, int idx) {
    return &snn->neurons[idx];
}

void printSNN(SNN* snn) {
    printf("\n\t~~ / SNN \\ ~~\n");
    printf("snn # of inputs: %d\n", snn->n_inputs);
    printf("snn # of neurons: %d\n", snn->n_neurons);
    printf("snn # of synapses: %d\n", snn->n_synapses);
    printf("snn # of spikes: %d\n\n", snn->n_spikes);
    printf("Neurons:\n");
    
    for (int ni = 0; ni < snn->n_neurons; ni++) {
        printNeuron(&snn->neurons[ni], ni);
        printSynapses(*snn->neurons[ni].synapses, snn->neurons[ni].n_synapses);
    }
}