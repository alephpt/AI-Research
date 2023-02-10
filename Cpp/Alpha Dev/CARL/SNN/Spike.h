#pragma once
#include "../Types/Types.h"

struct Spike{
	bool fired;																			// determines if a Spike has been fired
	fscalar amplitude;																	// defined by the membrane_potential - V_rest;
	double timestamp;																	// determines when a spike was fired
};


void initSpike(Spike* s, fscalar a);
Spike* createNewSpike(Spike* s, fscalar a);
void printSpike(Spike* s, int n);
// bool isFired();																		// determines if a spike was fired
// void fire();																			// sets fired to true
// void reset();																		// sets fired to false
