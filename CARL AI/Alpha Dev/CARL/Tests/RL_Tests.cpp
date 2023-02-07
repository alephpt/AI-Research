#include "RL_Tests.h"

void testRLQLearning() {
	QLearning q(10, 10);
	q.Train(100, 0.1f, 0.9f, 0.1f);
	return;
}