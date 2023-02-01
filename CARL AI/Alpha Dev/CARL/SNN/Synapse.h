#pragma once
#include "SNNNeuron.h"

typedef struct SNNNeuron SNNNeuron;

struct Synapse{
	int index;
	float weight;																		// the strength of the connection
	float delay;																		// time between the synaptic input and output
	float decay;																		// contributes to greater resistance, higher threshold and rest && higher weight
	int n_spike_times;																    // number of spike times
	double* spike_times;																// list of time instances when spikes occured
	SNNNeuron* pre_neuron;																	// the neuron that the current synapse is connected from
	SNNNeuron* post_neuron;																// the neuron the current synapse is connected to
};


// /// /// Synapse Functions /// /// //
void initSynapse(Synapse* s, float weight, float delay, float decay, SNNNeuron* pre_neuron, SNNNeuron* post_neuron);
Synapse* createNewSynapse(Synapse* s, int idx, float w, float del, float dec, SNNNeuron* pre, SNNNeuron* post);
void printSynapses(SNNNeuron* n);
// void setWeight(float weight);														// sets the weight of a synapse
// void update()																		// updates the synpases output based on the input spike and weight

