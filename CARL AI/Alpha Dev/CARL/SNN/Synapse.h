#pragma once
#include "Neuron.h"

typedef struct Neuron Neuron;

struct Synapse{
	int index;
	float weight;																		// the strength of the connection
	float delay;																		// time between the synaptic input and output
	int n_spike_times;																    // number of spike times
	double* spike_times;																// list of time instances when spikes occured
	Neuron* pre_neuron;																	// the neuron that is sending the signal
	Neuron* post_neuron;																// the neuron that is receiving the signal
};


// /// /// Synapse Functions /// /// //
void initSynapse(Synapse* s, float weight, float delay, Neuron* pre_neuron, Neuron* post_neuron);
Synapse* createNewSynapse(Synapse* s, int idx, float w, float d, Neuron* pre, Neuron* post);
void printSynapses(Synapse* ss, int n_s);
void connectNeurons(int neuron_a, int neuron_b);
// void setWeight(float weight);														// sets the weight of a synapse
// void update()																		// updates the synpases output based on the input spike and weight

