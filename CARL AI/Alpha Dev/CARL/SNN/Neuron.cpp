#include "Neuron.h"
#include <math.h>

float membranePotential(Membrane m) {
    return m.V_rest + (m.V_threshold - m.V_rest) * (1.0f - expf(-m.t / (m.resistance * m.capacitance)));
}

Neuron* initNeuron(Neuron* n, int idx, float* w, float b, float in, float out, float d, float mem_p, float t, 
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
    
    n->membrane = *createNewMembrane(m, 0.0f, 0.0f, 0.0f, 1.0f, t);

    for(int i=0; i<n_syn; i++) {
        n->synapses[i] = *createNewSynapse(&syn[i], 0.0f, 0.0f, NULL, NULL);
    }

    for(int i=0; i<n_sp; i++) {
        n->spikes[i] = *createNewSpike(&sp[i]);
    }

    return n;
}

void initMembrane(Membrane* m, float capacitance, float resistance, float V_rest, float V_threshold, float t) {
    m->capacitance = capacitance;
    m->resistance = resistance;
    m->V_rest = V_rest;
    m->V_threshold = V_threshold;
    m->t = t;
}

Membrane* createNewMembrane(Membrane* m, float c, float r, float V_rest, float V_thresh, float t) {
    initMembrane(m, c, r, V_rest, V_thresh, t);
    return m;
}

void initSynapse(Synapse* s, float weight, float delay, Neuron* pre_neuron, Neuron* post_neuron) {
    s->weight = weight;
    s->delay = delay;
    s->n_spike_times = 0;
    s->spike_times = NULL;
    s->pre_neuron = pre_neuron;
    s->post_neuron = post_neuron;
}

Synapse* createNewSynapse(Synapse* s, float w, float d, Neuron* pre, Neuron* post) {
    initSynapse(s, w, d, pre, post);
    return s;
}

void initSpike(Spike* s) {
    s->fired = false;
    s->amplitude = 0;
    s->timestamp = 0;
}

Spike* createNewSpike(Spike* s) {
    initSpike(s);
    return s;
}