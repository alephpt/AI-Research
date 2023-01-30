#pragma once
#include "Neuron.h"

typedef struct Neuron Neuron;

struct Synapse{
	int index;
	float weight;																		// the strength of the connection
	float delay;																		// time between the synaptic input and output
	float decay;																		// contributes to greater resistance, higher threshold and rest && higher weight
	int n_spike_times;																    // number of spike times
	double* spike_times;																// list of time instances when spikes occured
	Neuron* pre_neuron;																	// the neuron that the current synapse is connected from
	Neuron* post_neuron;																// the neuron the current synapse is connected to
};


// /// /// Synapse Functions /// /// //
void initSynapse(Synapse* s, float weight, float delay, float decay, Neuron* pre_neuron, Neuron* post_neuron);
Synapse* createNewSynapse(Synapse* s, int idx, float w, float del, float dec, Neuron* pre, Neuron* post);
void printSynapses(Neuron* n);
void connectNeurons(int neuron_a, int neuron_b);
// void setWeight(float weight);														// sets the weight of a synapse
// void update()																		// updates the synpases output based on the input spike and weight

