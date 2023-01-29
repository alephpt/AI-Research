#pragma once


struct Membrane {
	float capacitance;																	// the ability to store a charge
	float resistance;																	// the opposition to the flow of current
	float V_rest;																		// potential of a neuron when it is resting
	float V_threshold;																	// potential at which a Neuron will fire
	double t;																			// time of which the membrane is being calculated
};


// /// /// Membrane Functions /// /// //
void initMembrane(Membrane* m, float capacitance, float resistance, float V_rest, float V_threshold);
Membrane* createNewMembrane(Membrane* m, float c, float r, float V_rest, float V_thresh);
void printMembrane(Membrane* m);
// getMembranePotential();																// returns the current membrane potential
// updateMembranePotential(float capacitance, float resistance, float inputCurrent);	// updates the potential
// isFired();																			// checks if the membrane potential has reached the firing threshold
// resetMembranePotential();															// resets the membrane potential to the resting potential
	