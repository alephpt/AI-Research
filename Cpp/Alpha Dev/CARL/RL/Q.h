#pragma once
#include "../Types/Types.h"
#include "Environment.h"


class QLearning {
public:
	QLearning(int x_size_, int y_size_, State* state);
	~QLearning();

	Environment* getEnvironment();
	ftensor3d getQValues();
	void Train(int, fscalar, fscalar, fscalar);

private:
	Environment env;
	ftensor3d q_values;
	int x_size;
	int y_size;
};