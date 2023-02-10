#pragma once
#include "../Types/Types.h"

typedef struct {
	fscalar capacitance;																	// the ability to store a charge
	fscalar resistance;																	// the opposition to the flow of current
	fscalar V_rest;																		// potential of a neuron when it is resting
	fscalar V_threshold;																	// potential at which a Neuron will fire
	double t;																			// time of which the membrane is being calculated
} Membrane;


// /// /// Membrane Functions /// /// //
void initMembrane(Membrane* m, fscalar capacitance, fscalar resistance, fscalar V_rest, fscalar V_threshold);
extern Membrane* createNewMembrane(Membrane* m, fscalar c, fscalar r, fscalar V_rest, fscalar V_thresh);
void printMembrane(Membrane* m);
// getMembranePotential();																// returns the current membrane potential
// updateMembranePotential(fscalar capacitance, fscalar resistance, fscalar inputCurrent);	// updates the potential
// isFired();																			// checks if the membrane potential has reached the firing threshold
// resetMembranePotential();															// resets the membrane potential to the resting potential
	