#pragma once
#include "../Types/Activation.h"
#include "../Types/General.h"
#include "Spike.h"
#include "Synapse.h"
#include "Membrane.h"

typedef struct Membrane Membrane;
typedef struct Synapse Synapse;
typedef struct Spike Spike;

struct Neuron {
	int index;																			// used for topology
	float input;																		// stores the input of the neuron before the function is applied
	float output;																		// stores the output of the neuron after the function is applied
	float* weights;																		// stores connection weight between current neuron and other neurons in the network
	float bias;																			// can be added to the input before passing it through the activation function
	float delta;																		// used for backpropagation
	float membrane_potential;															// potential of a neuron firing
	Membrane membrane;																	// used to determine the membrane potential
	int n_synapses;																		// number of Synapses
	Synapse* synapses;																	// connects Neurons to other Neurons to transmit spikes;
	int n_spikes;																		// number of neural spikes
	Spike* spikes;																		// used to track neural activity over time
	Activation activation_type;															// determins the activation type to calculate the weights and bias' of a Neuron
};


// /// /// Neuron Functions /// /// //
void printNeuron(Neuron* n, int n_i, int n_w);
Neuron* initNeuron(Neuron* n, int idx, float* w, float b, float in, float out, float d, float mem_p, 
                   Membrane* m, int n_syn, Synapse* syn, int n_sp, Spike* sp, Activation activation_type);
float membranePotential(Membrane m);
// addInputSynapse(Synapse* synapse);													// add input synapse to Neuron
// addOutputSynapse(Synapse* synapse);													// add output synapse to Neuron
// void update();																		// updates the neurons membrane potential and generates a spike if the threshold is reached
// bool isFired();;																		// checks if the neuron has fired a spike in the current time step

