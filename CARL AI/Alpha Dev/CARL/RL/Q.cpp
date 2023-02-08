#include "Q.h"
#include <random>


QLearning::QLearning(int x_size_, int y_size_, State* state) :
	env(x_size_, y_size_, state), x_size(x_size_), y_size(y_size_)
{
	q_values = ftensor3d(y_size, fmatrix(x_size, vector<fscalar>(4)));
}

QLearning::~QLearning() { q_values.clear(); }

Environment* QLearning::getEnvironment()
{
	return &env;
}

ftensor3d QLearning::getQValues()
{
	return q_values;
}

void QLearning::Train(int n_episodes, fscalar r_learning, fscalar f_discount, fscalar epsilon) {
	std::random_device randev;
	std::mt19937 gen(randev());
	std::uniform_real_distribution<fscalar> dis(0.0f, 1.0f);

	for (int ep = 0; ep < n_episodes; ++ep) {
		State current_state = *env.getStartState();
		fscalar total_reward = 0;

		while (true) {
			Action action;

			if (dis(gen) < epsilon) {
				std::uniform_int_distribution<> action_dis(0, 3);
				action = (Action)(action_dis(gen));
			}
			else {
				int best_action = 0;
				fscalar best_value = q_values[current_state.y][current_state.x][0];

				for (int i = 1; i < 4; ++i) {
					fscalar v = q_values[current_state.y][current_state.x][i];

					if (v > best_value) {
						best_value = v;
						best_action = i;
					}
				}

				action = (Action)(best_action);
			}

			State next_state = env.getNextState(&current_state, &action);
			fscalar reward = env.GetReward(&current_state, &next_state);
			total_reward += reward;

			fscalar best_future_value = q_values[next_state.y][next_state.x][0];

			for (int i = 1; i < 4; ++i) {
				fscalar v = q_values[next_state.y][next_state.x][i];

				if (v > best_future_value) {
					best_future_value = v;
				}
			}

			q_values[current_state.y][current_state.x][(int)(action)] = (1.0f - r_learning) * q_values[current_state.y][current_state.x][(int)(action)] + r_learning * (reward + f_discount * best_future_value);

			if ((next_state.x == (x_size - 1)) && (next_state.y == (y_size - 1))) {
				break;
			}

			current_state = next_state;
			env.setStartState(&current_state);
		}

		if (ep % (n_episodes / 10) == 0) {
			printf("QLearning Episode %d: \nTotal Reward = %f\n", ep, total_reward);
		}
	}
}