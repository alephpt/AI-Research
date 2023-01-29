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

int synapseLimit(int n_neurons, float density) {
    return (int)((float)(n_neurons * (n_neurons - 1) / 2) * density);
}

void initSNN(SNN* snn, int n_in, int n_n, int n_syn, int n_sp, Activation activation_type, double t) {
    snn->n_inputs = n_in;
    snn->n_neurons = n_n;
    snn->n_synapses = n_syn;
    snn->n_spikes = n_sp;
    snn->neurons = new Neuron[n_n];
    snn->t = t;

    for (int n_i = 0; n_i < n_n; n_i++) {
        initNeuron(&snn->neurons[n_i], n_i, new float[n_in] , 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, t,
                   new Membrane, n_syn, new Synapse[n_syn], n_sp, new Spike[n_sp], activation_type);
    }
}

void printSNN(SNN* snn) {
    printf("\t~~ / SNN \\ ~~\n");
    printf("snn # of inputs: %d\n", snn->n_inputs);
    printf("snn # of neurons: %d\n", snn->n_neurons);
    printf("snn # of synapses: %d\n", snn->n_synapses);
    printf("snn # of spikes: %d\n\n", snn->n_spikes);
    printf("Neurons:\n");
    
    for (int n_i = 0; n_i < snn->n_neurons; n_i++) {
        printf("\t\t\t-- Neuron %d --\n", n_i);
        printf("activation_type:\t\t%s\n", getActivationString.at(snn->neurons[n_i].activation_type).c_str());
        printf("index: \t\t\t\t%d\n", snn->neurons[n_i].index);
        printf("input: \t\t\t\t%f\n", snn->neurons[n_i].input);
        printf("output: \t\t\t%f\n", snn->neurons[n_i].output);
        printf("weights:\n  ");
        for (int n_w = 0; n_w < snn->n_inputs; n_w++) {
            printf("- %d: %f -", n_w, snn->neurons[n_i].weights[n_w]);
            if ((n_w + 1) % 5 == 0 && n_w != 0 || n_w == snn->n_inputs - 1) {
                printf("\n");
            }
        }
        printf("bias: \t\t\t\t%f\n", snn->neurons[n_i].bias);
        printf("delta: \t\t\t\t%f\n", snn->neurons[n_i].delta);
        printf("membrane_potential: \t\t%f\n", snn->neurons[n_i].bias);
        printf("Membrane:\n");
        printf("\tcapacitance: \t\t%f\n", snn->neurons[n_i].membrane.capacitance);
        printf("\tresistance: \t\t%f\n", snn->neurons[n_i].membrane.resistance);
        printf("\tV_rest: \t\t%f\n", snn->neurons[n_i].membrane.V_rest);
        printf("\tV_threshold: \t\t%f\n", snn->neurons[n_i].membrane.V_threshold);
        printf("\tt: \t\t\t%lf\n", snn->neurons[n_i].membrane.t);
        printf("n_synapses: %d\n", snn->neurons[n_i].n_synapses);
        for (int n_s = 0; n_s < snn->neurons[n_i].n_synapses; n_s++) {
            printf("\t- Synapse %d:\n", n_s);
            printf("\t\tweight: \t%f\n", snn->neurons[n_i].synapses[n_s].weight);
            printf("\t\tdelay: \t\t%f\n", snn->neurons[n_i].synapses[n_s].delay);
            printf("\t\tn_spike_time: \t%d\n", snn->neurons[n_i].synapses[n_s].n_spike_times);
            printf("\t\t\t");
            for (int nst = 0; nst < snn->neurons[n_i].synapses[n_s].n_spike_times; nst++) {
                if ((nst + 1) % 9 == 0) { printf("\t\t"); }
                printf("%f", snn->neurons[n_i].synapses[n_s].spike_times[nst]);
                if (nst != snn->neurons[n_i].synapses[n_s].n_spike_times - 1) {
                    printf(", ");
                }
                if ((nst + 1) % 9 == 0 || nst == snn->neurons[n_i].synapses[n_s].n_spike_times - 1) { printf("\n"); }
            } 
            int preidx = 0;
            int postidx = 0;
            if (snn->neurons[n_i].synapses[n_s].pre_neuron != NULL) {
                preidx = snn->neurons[n_i].synapses[n_s].pre_neuron->index;
            }
            if (snn->neurons[n_i].synapses[n_s].post_neuron != NULL) {
                preidx = snn->neurons[n_i].synapses[n_s].post_neuron->index;
            }
            printf("\t\tpre_neuron: \t%d\n", preidx);
            printf("\t\tpost_neuron: \t%d\n", postidx);
        }
        printf("n_spikes: %d\n", snn->neurons[n_i].n_spikes);
        for (int n_s = 0; n_s < snn->neurons[n_i].n_spikes; n_s++) {
            printf("\t- Spike %d:\n", n_s);
            printf("\t\tfired: \t\t%s\n", snn->neurons[n_i].spikes[n_s].fired ? "true" : "false");
            printf("\t\ttime: \t\t%lf\n", snn->neurons[n_i].spikes[n_s].timestamp);
            printf("\t\tamplitude: \t%f\n", snn->neurons[n_i].spikes[n_s].amplitude);
        }
        printf("\n");
    }
}