#pragma once
#include "../Types/Types.h"
#include "Environment.h"


class QLearning {
public:
	QLearning(int x_size_, int y_size_, State* state);
	~QLearning();

	Environment* getEnvironment();
	tensorf3d getQValues();
	void Train(int, float, float, float);

private:
	Environment env;
	tensorf3d q_values;
	int x_size;
	int y_size;
};