#pragma once
#include "Environment.h"
#include "Action.h"
#include "State.h"
#include "../Types/Types.h"

class Policy {
public:
	Policy(ftensor3d);
	
	Action ChooseAction(const State*, fscalar);
	void Rollout(int, Environment*, fscalar);

private:
	ftensor3d q_values;
};