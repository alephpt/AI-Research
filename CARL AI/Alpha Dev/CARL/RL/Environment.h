#pragma once
#include "State.h"
#include <math.h>
#include "Action.h"

class Environment {
public:
	Environment(int x_size_, int y_size_, State* state);
		
	State* getStartState();
	void setStartState(State*);
	State getNextState(const State* current_state, const Action* action);
	float GetReward(const State* current_state, const State* next_state);

private:
	State* start_state;
	int x_size;
	int y_size;
};