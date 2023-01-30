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

void printSynapses(Synapse* ss, int n_s) {
    printf("n_synapses: %d\n", n_s);
    for (int n = 0; n < n_s; n++) {
        printf("\t- Synapse %d:\n", n);
        printf("\t\tindex: \t%f\n", ss[n].index);
        printf("\t\tweight: \t%f\n", ss[n].weight);
        printf("\t\tdelay: \t\t%f\n", ss[n].delay);
        printf("\t\tn_spike_time: \t%d\n", ss[n].n_spike_times);
        printf("\t\t\t");
        for (int nst = 0; nst < ss[n].n_spike_times; nst++) {
            if ((nst + 1) % 9 == 0) { printf("\t\t"); }
            printf("%.9lf", ss[n].spike_times[nst]);
            if (nst != ss[n].n_spike_times - 1) {
                printf(", ");
            }
            if ((nst + 1) % 9 == 0 || nst == ss[n].n_spike_times - 1) { printf("\n"); }
        } 
        int preidx = 0;
        int postidx = 0;
        if (ss[n].pre_neuron != NULL) { preidx = ss[n].pre_neuron->index; }
        if (ss[n].post_neuron != NULL) { preidx = ss[n].post_neuron->index; }
        printf("\t\tpre_neuron: \t%d\n", preidx);
        printf("\t\tpost_neuron: \t%d\n", postidx);
    }
}
