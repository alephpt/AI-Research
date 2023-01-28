#include "Neuron.h"
#include <math.h>

float membranePotential(float inputCurrent, Membrane membrane) {
    return membrane.V_rest + (membrane.V_threshold - membrane.V_rest) * (1.0f - expf(-membrane.t / (membrane.resistance * membrane.capacitance)));
}

void initSynapse(Synapse* synapse, float weight, float delay, Neuron* pre_neuron, Neuron* post_neuron) {
    synapse->weight = weight;
    synapse->delay = delay;
    synapse->n_spike_times = 0;
    synapse->spike_times = NULL;
    synapse->pre_neuron = pre_neuron;
    synapse->post_neuron = post_neuron;
}

Synapse* createNewSynapse(Synapse* newSynapse, float w, float d, Neuron* pre, Neuron* post) {
    initSynapse(newSynapse, w, d, pre, post);
    return newSynapse;
}

void initSpike(Spike* spike) {
    spike->fired = false;
    spike->amplitude = 0;
    spike->timestamp = 0;
}

Spike* createNewSpike(Spike* spike) {
    initSpike(spike);
    return spike;
}