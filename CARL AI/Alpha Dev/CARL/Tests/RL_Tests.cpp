#include "RL_Tests.h"

void testRLQLearning() {
	QLearning q(10, 10, new State);
	q.Train(1000, 0.1f, 0.9f, 0.1f);
	Policy p(q.getQValues());
	p.Rollout(1000, q.getEnvironment(), 0.1f);
	return;
}