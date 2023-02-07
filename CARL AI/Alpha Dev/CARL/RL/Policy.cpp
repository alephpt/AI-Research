#include "Policy.h"
#include <cstdlib>

Policy::Policy(tensorf3d q_values_) { q_values = q_values_; }

Action Policy::ChooseAction(const State* state, float epsilon) {
	 if (rand() < (epsilon * RAND_MAX)) {
		return (Action)(rand() % 4);
	 } else {
		 int best_action = 0;
		 float best_value = q_values[state->y][state->x][0];

		 for (int i = 1; i < 4; ++i) {
			 float value = q_values[state->y][state->x][i];

			 if (value > best_value) {
				 best_value = value;
				 best_action = i;
			 }
		 }

		 return (Action)(best_action);
	}
}

void Policy::Rollout(int n_episodes, Environment* env, float epsilon)
{
	for (int ep = 0; ep < n_episodes; ++ep) {
		State* current_state = env->getStartState();
		float total_reward = 0.0;

		while (true) {
			Action action = ChooseAction(current_state, epsilon);
			State next_state = env->getNextState(current_state, &action);
			float reward = env->GetReward(current_state, &next_state);
			total_reward += reward;

			if (next_state.x == 9 && next_state.y == 9) {
				break;
			}

			current_state = &next_state;
		}


		if (ep % (n_episodes / 10) == 0) {
			printf("Policy Episode %d: \nTotal Reward = %f\n", ep, total_reward);
		}
	}
}
