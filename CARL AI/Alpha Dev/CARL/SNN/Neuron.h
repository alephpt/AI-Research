#pragma once
#include "../Types/Activation.h"

typedef struct Membrane {
	float capacitance;																	// the ability to store a charge
	float resistance;																	// the opposition to the flow of current
	float V_rest;																		// potential of a neuron when it is resting
	float V_threshold;																	// potential at which a Neuron will fire
	float t;																			// time of which the membrane is being calculated
};

typedef struct Synapse {
	float weight;																		// the strength of the connection
	float delay;																		// time between the synaptic input and output
	float n_spike_times;																// number of spike times
	float* spike_times;																	// list of time instances when spikes occured
	Neuron* pre_neuron;																	// the neuron that is sending the signal
	Neuron* post_neuron;																// the neuron that is receiving the signal
} Synapse;

typedef struct Spike {
	bool fired;																			// determines if a Spike has been fired
	float amplitude;																	// defined by the membrane_potential - V_rest;
	int timestamp;																		// determines when a spike was fired
} Spike;

typedef struct Neuron {
	int index;																			// used for topology
	float* weights;																		// stores connection weight between current neuron and other neurons in the network
	float bias;																			// can be added to the input before passing it through the activation function
	float output;																		// stores the output of the neuron after the function is applied
	float delta;																		// used for backpropagation
	float memberane_potential;															// potential of a neuron firing
	Membrane membrane;																	// used to determine the membrane potential
	int n_synapses;																		// number of Synapses
	Synapse* synapses;																	// connects Neurons to other Neurons to transmit spikes;
	int n_spikes;																		// number of neural spikes
	Spike* spikes;																		// used to track neural activity over time
	Activation activation_type;															// determins the activation type to calculate the weights and bias' of a Neuron
} Neuron;



// /// /// Membrane Functions /// /// //
// getMembranePotential();																// returns the current membrane potential
// updateMembranePotential(float capacitance, float resistance, float inputCurrent);	// updates the potential
// isFired();																			// checks if the membrane potential has reached the firing threshold
// resetMembranePotential();															// resets the membrane potential to the resting potential
	
// /// /// Neuron Functions /// /// //
float membranePotential(float inputCurrent, Membrane membrane);
// addInputSynapse(Synapse* synapse);													// add input synapse to Neuron
// addOutputSynapse(Synapse* synapse);													// add output synapse to Neuron
// void update();																		// updates the neurons membrane potential and generates a spike if the threshold is reached
// bool isFired();;																		// checks if the neuron has fired a spike in the current time step

// /// /// Synapse Functions /// /// //
// void setWeight(float weight);														// sets the weight of a synapse
// void update()																		// updates the synpases output based on the input spike and weight

// /// /// Spike Functions /// /// //
// bool isFired();																		// determines if a spike was fired
// void fire();																			// sets fired to true
// void reset();																		// sets fired to false