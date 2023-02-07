#pragma once
#include "../Types/Types.h"
#include <algorithm>
#include <random>
#include <math.h>
#include <stdio.h>

// need to include a set of inputs
// need to include a set of accuracy templates given a set of inputs

// Define Environment
  // State Space, Action Space, Reward Function and Constraints
// Select RL Algorithm
   // Q-Learning, SARSA, Actor_Critic
// Initialize Agent 
   // Initial Policy and Parameters - Q-Table or NN Weights
// Take Actions
// Update Policies
// Repeat prior 2 steps until convergences
// Test final policy

typedef struct State {
/*
	int GAN_Discriminator_Cost;
	int GAN_Discriminator_Error;
	int GAN_Generator_Cost;
	int GAN_Generator_Error;
	int CNN_Convolution_Cost;
	int CNN_Pooling_Cost;
	int CNN_Neural_Error;
	int RNN_Network_Cost;
	int RNN_Network_Error;
	*/
	int x;
	int y;
} State;

enum class Action {
	Up,
	Down,
	Left,
	Right,
};

class Environment {
public:
	Environment (int x_size_, int y_size_) : 
		x_size(x_size_), y_size(y_size_) {}

	State getNextState(const State& current_state, const Action& action) const {
		int x = current_state.x;
		int y = current_state.y;

		switch (action) {
			case Action::Up:
				y = std::max(y - 1, 0);
				break;
			case Action::Left:
				x = std::max(x - 1, 0);
				break;
			case Action::Right:
				x = std::min(x + 1, x_size - 1);
				break;
			case Action::Down:
				y = std::min(y + 1, y_size - 1);
				break;
		}

		return State{ x, y };
	}

	float GetReward(const State& current_state, const State& next_state) const {
		float x_diff = (float)(next_state.x - current_state.x);
		float y_diff = (float)(next_state.y - current_state.y);
		return -sqrtf(x_diff * x_diff + y_diff * y_diff);
	}

private:
	int x_size;
	int y_size;
};

class QLearning {
public: 
	QLearning(int x_size_, int y_size_) : 
		env(x_size_, y_size_), x_size(x_size_), y_size(y_size_)
	{
		q_values = ftensor3d(y_size, fmatrix(x_size, vector<float>(4)));
	}

	void Train(int n_episodes, float r_learning, float f_discount, float epsilon) {
		std::random_device randev;
		std::mt19937 gen(randev());
		std::uniform_real_distribution<float> dis(0.0f, 1.0f);

		for (int ep = 0; ep < n_episodes; ++ep) {
			State current_state{ 0, 0 };
			float total_reward = 0;

			while (true) {
				Action action;

				if (dis(gen) < epsilon) {
					std::uniform_int_distribution<> action_dis(0, 3);
					action = (Action)(action_dis(gen));
				}
				else {
					int best_action = 0;
					float best_value = q_values[current_state.y][current_state.y][0];

					for (int i = 1; i < 4; ++i) {
						float v = q_values[current_state.y][current_state.x][i];

						if (v > best_value) {
							best_value = v;
							best_action = i;
						}
					}

					action = (Action)(best_action);
				}

				State next_state = env.getNextState(current_state, action);
				float reward = env.GetReward(current_state, next_state);
				total_reward += reward;

				float best_future_value = q_values[next_state.y][next_state.x][0];

				for (int i = 1; i < 4; ++i) {
					float v = q_values[next_state.y][next_state.x][i];

					if (v > best_future_value) {
						best_future_value = v;
					}
				}

				q_values[current_state.y][current_state.x][(int)(action)] = (1.0f - r_learning) * q_values[current_state.y][current_state.x][(int)(action)] + r_learning * (reward + f_discount * best_future_value);

				if ((next_state.x = (x_size - 1)) && (next_state.y == (y_size - 1))) {
					break;
				}

				current_state = next_state;
			}

			printf("Episode %d: Total Reward = %f\n", ep, total_reward);
		}
	}

private:
	Environment env;
	ftensor3d q_values;
	int x_size;
	int y_size;
};