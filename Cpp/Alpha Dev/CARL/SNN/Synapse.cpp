#include "Synapse.h"
#include "../Types/General.h"
#include "SNN.h"

void initSynapse(Synapse* s, fscalar weight, fscalar delay, fscalar decay, SNNNeuron* pre_neuron, SNNNeuron* post_neuron) {
    s->weight = weight;
    s->delay = delay;
    s->decay = decay;
    s->n_spike_times = 1;
    s->spike_times = new double[s->n_spike_times];
    s->spike_times[0] = getTime();
    s->pre_neuron = pre_neuron;
    s->post_neuron = post_neuron;
}

Synapse* createNewSynapse(Synapse* s, int idx, fscalar w, fscalar del, fscalar dec, SNNNeuron* pre, SNNNeuron* post) {
    initSynapse(s, w, del, dec, pre, post);
    s->index = idx;
    return s;
}

void connectNeurons(SNN* snn, Synapse* s, int neuron_a, int neuron_b) {
    s->pre_neuron = &snn->neurons[neuron_a];
    s->post_neuron = &snn->neurons[neuron_b];
}

void printSynapses(SNNNeuron* n) {
    Synapse** ss = n->synapses;
    int n_s = n->n_synapses;
    printf("n_synapses: %d\n", n_s);
    for (int si = 0; si < n_s; si++) {
        int preidx = 0;
        int postidx = 0;
        if (ss[si]->pre_neuron != NULL) { preidx = ss[si]->pre_neuron->index; }
        if (ss[si]->post_neuron != NULL) { postidx = ss[si]->post_neuron->index; }
        
        printf("\t- Synapse %d:\t", si);
        printf("\tindex: \t\t%d", ss[si]->index);
        printf("\t\tpre_neuron: \t%d", preidx);
        printf("\t\tpost_neuron: \t%d\n", postidx);
        printf("\t\t\t\tweight: \t%f", ss[si]->weight);
        printf("\tdelay: \t\t%f", ss[si]->delay);
        printf("\tdecay: \t\t%f\n", ss[si]->decay);
        printf("\t\t\t\tn_spike_time: \t%d\n\t\t\t\t\t\t", ss[si]->n_spike_times);
        printf("[ ");
        for (int nst = 0; nst < ss[si]->n_spike_times; nst++) {
            printf("%.9lf", ss[si]->spike_times[nst]);
            if (nst != ss[si]->n_spike_times - 1) { printf(", "); }
            if (nst == ss[si]->n_spike_times - 1) { printf(" ]\n"); }
            else  if ((nst + 1) % 9 == 0) { printf("\n\t\t\t\t\t\t"); }
        } 

    }
    printf("\n");
}
