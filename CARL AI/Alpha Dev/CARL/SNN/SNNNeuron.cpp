#include "SNNNeuron.h"
#include "Membrane.h"
#include "../Types/General.h"
#include <math.h>

 fscalar membranePotential(Membrane m) {
    return m.V_rest + (m.V_threshold - m.V_rest) * (1.0f - expf((fscalar)(- m.t) / (m.resistance * m.capacitance)));
}

SNNNeuron* initNeuron(Activation activation_type, SNNNeuron* n, Membrane* m, Spike* sp, int n_sp, 
                    fscalar* w, fscalar b, int n_in, fscalar* in, int n_out, fscalar* out, fscalar d, fscalar mem_p) {
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
    n->activation_type = activation_type;
    n->membrane = *createNewMembrane(m, 0.0f, 0.0f, 0.0f, 1.0f);

    for (int i = 0; i < n_in; i++) { n->inputs[i] = 0.0f; n->weights[i] = 0.0f; }
    for (int i = 0; i < n_out; i++) { n->outputs[i] = 0.0f; }



    for(int i = 0; i < n_sp; i++) {
        n->spikes[i] = *createNewSpike(&sp[i], 0.0f);
    }

    return n;
}

void printNeuron(SNNNeuron* n, int n_i) {
    printf("\n\t\t\t-- Neuron %d --\n", n_i);
    printf("activation_type:\t\t%s\n", activationString[n->activation_type].c_str());
    printf("index: \t\t\t\t%d\n", n->index);
    printf("inputs: \t\t\t%d\n\t\t\t\t[ ", n->n_inputs);
    for (int i = 0; i < n->n_inputs; i++) {
        printf("%d: %f", i, n->inputs[i]);
        if (i == n->n_inputs - 1) {
            printf(" ]\n");
        }
        else if ((i + 1) % 5 == 0 && i != 0) {
            printf("\t\t\t\t\n");
        }
        else {
            printf(", ");
        }
    }
    printf("outputs: \t\t\t%d\n\t\t\t\t[ ", n->n_outputs);
    for (int i = 0; i < n->n_outputs; i++) {
        printf("%d: %f", i, n->outputs[i]);
        if (i == n->n_outputs - 1) {
            printf(" ]\n");
        }
        else if ((i + 1) % 5 == 0 && i != 0) {
            printf("\t\t\t\t\n");
        }
        else {
            printf(", ");
        }
    }
    printf("weights:\t\t\t[ ");
    for (int w = 0; w < n->n_inputs; w++) {
        printf("%d: %f", w, n->weights[w]);
        if (w == n->n_inputs - 1) {
            printf(" ]\n");
        } else
        if ((w + 1) % 5 == 0 && w != 0) {
            printf("\t\t\t\t\n");
        }
        else {
            printf(", ");
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