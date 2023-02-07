#pragma once
#include "Environment.h"
#include "Action.h"
#include "State.h"
#include "../Types/Types.h"

class Policy {
public:
	Policy(tensorf3d);
	
	Action ChooseAction(const State*, float);
	void Rollout(int, Environment*, float);

private:
	tensorf3d q_values;
};