#include "Synapse.h"
#include "../Types/General.h"
#include "SNN.h"

void initSynapse(Synapse* s, float weight, float delay, Neuron* pre_neuron, Neuron* post_neuron) {
    s->weight = weight;
    s->delay = delay;
    s->n_spike_times = 1;
    s->spike_times = new double[s->n_spike_times];
    s->spike_times[0] = getTime();
    s->pre_neuron = pre_neuron;
    s->post_neuron = post_neuron;
}

Synapse* createNewSynapse(Synapse* s, int idx, float w, float d, Neuron* pre, Neuron* post) {
    initSynapse(s, w, d, pre, post);
    s->index = idx;
    return s;
}

void connectNeurons(SNN* snn, Synapse* s, int neuron_a, int neuron_b) {
    s->pre_neuron = &snn->neurons[neuron_a];
    s->post_neuron = &snn->neurons[neuron_b];
}

void printSynapses(Neuron* n) {
    Synapse** ss = n->synapses;
    int n_s = n->n_synapses;
    printf("n_synapses: %d\n", n_s);
    for (int si = 0; si < n_s; si++) {
        printf("\t- Synapse %d:\n", si);
        printf("\t\tindex: \t%f", ss[si]->index);
        printf("\t\tweight: \t%f", ss[si]->weight);
        printf("\t\tdelay: \t\t%f\n", ss[si]->delay);
        printf("\t\tn_spike_time: \t%d\n", ss[si]->n_spike_times);
        printf("\t\t\t");
        for (int nst = 0; nst < ss[si]->n_spike_times; nst++) {
            if ((nst + 1) % 9 == 0) { printf("\t\t"); }
            printf("%.9lf", ss[si]->spike_times[nst]);
            if (nst != ss[si]->n_spike_times - 1) {
                printf(", ");
            }
            if ((nst + 1) % 9 == 0 || nst == ss[si]->n_spike_times - 1) { printf("\n"); }
        } 
        int preidx = 0;
        int postidx = 0;
        if (ss[si]->pre_neuron != NULL) { preidx = ss[si]->pre_neuron->index; }
        if (ss[si]->post_neuron != NULL) { postidx = ss[si]->post_neuron->index; }
        printf("\t\tpre_neuron: \t%d", preidx);
        printf("\t\tpost_neuron: \t%d\n", postidx);
    }
}
