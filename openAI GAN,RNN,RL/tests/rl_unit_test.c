#include "./RL.h"

void test_RL() {
    int n_states = 2;
    int n_actions = 3;
    RL* rl = createRL(n_states, n_actions);

    // Create test data
    float newQ[n_states][n_actions] = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6}
    };
    float newPolicy[n_states][n_actions] = {
        {0.6, 0.5, 0.4},
        {0.3, 0.2, 0.1}
    };
    float newValue[n_states] = { 0.5, 0.6 };

    // Test updateQ
    updateQ(rl, newQ);
    for (int i = 0; i < n_states; i++) {
        for (int j = 0; j < n_actions; j++) {
            assert(rl->q[i][j] == newQ[i][j]);
        }
    }

    // Test updatePolicy
    updateNewPolicy(rl, newPolicy);
    for (int i = 0; i < n_states; i++) {
        for (int j = 0; j < n_actions; j++) {
            assert(rl->policy[i][j] == newPolicy[i][j]);
        }
    }

    // Test updateValue
    updateValue(rl, newValue);
    for (int i = 0; i < n_states; i++) {
        assert(rl->value[i] == newValue[i]);
    }
    destroyRL(rl);
}

int main() {
    test_RL();
    printf("All tests passed!");
    return 0;
}