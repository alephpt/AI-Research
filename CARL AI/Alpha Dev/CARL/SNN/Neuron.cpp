#include "Neuron.h"
#include <math.h>

float membranePotential(Membrane m) {
    return m.V_rest + (m.V_threshold - m.V_rest) * (1.0f - expf(-m.t / (m.resistance * m.capacitance)));
}

Neuron* initNeuron(Activation activation, Neuron* n, Membrane* m, Spike* sp, int n_sp, 
                    float* w, float b, int n_in, float* in, int n_out, float* out, float d, float mem_p) {
    n->weights = w;
    n->bias = b;
    n->n_inputs = n_in;
    n->inputs = in;
    n->n_outputs = n_out;
    n->outputs = out;
    n->delta = d;
    n->membrane_potential = mem_p;
    n->n_spikes = n_sp;
    n->spikes = sp;
    n->activation_type = activation;
    n->membrane = *createNewMembrane(m, 0.0f, 0.0f, 0.0f, 1.0f);

    for(int i = 0; i < n_sp; i++) {
        n->spikes[i] = *createNewSpike(&sp[i], 0.0f);
    }

    return n;
}

void printNeuron(Neuron* n, int n_i) {
    printf("\t\t\t-- Neuron %d --\n", n_i);
    printf("activation_type:\t\t%s\n", getActivationString.at(n->activation_type).c_str());
    printf("index: \t\t\t\t%d\n", n->index);
    printf("inputs: \t\t\t%d\n  ", n->n_inputs);
    for (int i = 0; i < n->n_inputs; i++) {
        printf("- %d: %f -", i, n->inputs[i]);
        if ((i + 1) % 5 == 0 && i != 0 || i == n->n_inputs - 1) {
            printf("\n");
        }
    }
    printf("outputs: \t\t\t%d\n  ", n->n_outputs);
    for (int i = 0; i < n->n_outputs; i++) {
        printf("- %d: %f -", i, n->outputs[i]);
        if ((i + 1) % 5 == 0 && i != 0 || i == n->n_outputs - 1) {
            printf("\n");
        }
    }
    printf("weights:\n  ");
    for (int w = 0; w < n_i - 1; w++) {
        printf("- %d: %f -", w, n->weights[w]);
        if ((w + 1) % 5 == 0 && w != 0 || w + 1 == n_i - 1) {
            printf("\n");
        }
    }
    printf("bias: \t\t\t\t%f\n", n->bias);
    printf("delta: \t\t\t\t%f\n", n->delta);
    printf("membrane_potential: \t\t%f\n", n->bias);
    printMembrane(&n->membrane);
    printf("n_spikes: %d\n", n->n_spikes);
    for (int n_s = 0; n_s < n->n_spikes; n_s++) {
        printSpike(&n->spikes[n_s], n_s);
    }
    printf("\n");
}