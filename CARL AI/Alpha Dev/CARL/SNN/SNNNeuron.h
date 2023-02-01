#pragma once
#include "../Types/Activation.h"
#include "Spike.h"
#include "Synapse.h"
#include "Membrane.h"

typedef struct Synapse Synapse;
typedef struct Spike Spike;

struct SNNNeuron {
	int index = 0;																		// used for topology
	int n_inputs = 0;
	float* inputs = new float;															// stores the input of the neuron before the function is applied
	int n_outputs = 0;
	float* outputs = new float;															// stores the output of the neuron after the function is applied
	float* weights = new float;															// stores connection weight between current neuron and other neurons in the network
	float bias = 0.0f;																	// can be added to the input before passing it through the activation function
	float delta = 0.0f;																	// used for backpropagation
	float membrane_potential = 0.0f;													// potential of a neuron firing
	Membrane membrane;																	// used to determine the membrane potential
	int n_synapses = 0;																	// number of Synapses
	Synapse** synapses = new Synapse*;													// connects Neurons to other Neurons to transmit spikes;
	int n_spikes = 0;																	// number of neural spikes
	Spike* spikes = new Spike;															// used to track neural activity over time
	Activation activation_type;															// determins the activation type to calculate the weights and bias' of a Neuron
};


// /// /// Neuron Functions /// /// //
float membranePotential(Membrane m);
void printNeuron(SNNNeuron* n, int n_i);
SNNNeuron* initNeuron(Activation activation, SNNNeuron* n, Membrane* m, Spike* sp, int n_sp,
                    float* w, float b, int n_in, float* in, int n_out, float* out, float d, float mem_p);
// addInputSynapse(Synapse* synapse);													// add input synapse to Neuron
// addOutputSynapse(Synapse* synapse);													// add output synapse to Neuron
// void update();																		// updates the neurons membrane potential and generates a spike if the threshold is reached
// bool isFired();;																		// checks if the neuron has fired a spike in the current time step

