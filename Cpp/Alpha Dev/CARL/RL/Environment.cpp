#include "Environment.h"
#include "../Types/Types.h"
#include <algorithm>

Environment::Environment(int x_size_, int y_size_, State* state) :
	x_size(x_size_), y_size(y_size_), start_state(state) {}

State* Environment::getStartState() {
	return start_state;
}

void Environment::setStartState(State* state)
{
	start_state = state;
}

State Environment::getNextState(const State* current_state, const Action* action) {
	int x = current_state->x;
	int y = current_state->y;

	switch (*action) {
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


// need to have sparsity to help the agent focus
// need to add immediacy to learn quickly and converge faster
// need to maintain feasibility so learning is easier
// rewards need consistency
// need clear objective
fscalar Environment::GetReward(const State* current_state, const State* next_state) {
	fscalar x_diff = (fscalar)(next_state->x - current_state->x);
	fscalar y_diff = (fscalar)(next_state->y - current_state->y);
	return -sqrtf(x_diff * x_diff + y_diff * y_diff);
}