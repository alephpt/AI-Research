#include "Neuron.h"
#include <math.h>


float membranePotential(Membrane m) {
    return m.V_rest + (m.V_threshold - m.V_rest) * (1.0f - expf(-m.t / (m.resistance * m.capacitance)));
}

Neuron* initNeuron(Neuron* n, int idx, float* w, float b, float in, float out, float d, float mem_p, 
                   Membrane* m, int n_syn, Synapse* syn, int n_sp, Spike* sp, Activation activation) {
    n->index = idx;
    n->weights = w;
    n->bias = b;
    n->input = in;
    n->output = out;
    n->delta = d;
    n->membrane_potential = mem_p;
    n->n_synapses = n_syn;
    n->synapses = syn;
    n->n_spikes = n_sp;
    n->spikes = sp;
    n->activation_type = activation;
    
    n->membrane = *createNewMembrane(m, 0.0f, 0.0f, 0.0f, 1.0f);

    for(int i=0; i< n_syn; i++) {
        n->synapses[i] = *createNewSynapse(&syn[i], 0.0f, 0.0f, NULL, NULL);
    }

    for(int i=0; i<n_sp; i++) {
        n->spikes[i] = *createNewSpike(&sp[i], 0.0f);
    }

    return n;
}

void printNeuron(Neuron* n, int n_i, int n_w) {
    printf("\t\t\t-- Neuron %d --\n", n_i);
    printf("activation_type:\t\t%s\n", getActivationString.at(n->activation_type).c_str());
    printf("index: \t\t\t\t%d\n", n->index);
    printf("input: \t\t\t\t%f\n", n->input);
    printf("output: \t\t\t%f\n", n->output);
    printf("weights:\n  ");
    for (int w = 0; w < n_w; w++) {
        printf("- %d: %f -", w, n->weights[w]);
        if ((w + 1) % 5 == 0 && w != 0 || w == n_w - 1) {
            printf("\n");
        }
    }
    printf("bias: \t\t\t\t%f\n", n->bias);
    printf("delta: \t\t\t\t%f\n", n->delta);
    printf("membrane_potential: \t\t%f\n", n->bias);
    printMembrane(&n->membrane);
    printSynapses(n->synapses, n->n_synapses);
    printf("n_spikes: %d\n", n->n_spikes);
    for (int n_s = 0; n_s < n->n_spikes; n_s++) {
        printSpike(&n->spikes[n_s], n_s);
    }
    printf("\n");
}