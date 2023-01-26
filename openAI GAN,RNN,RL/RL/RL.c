#include "RL.h"
#include "../CNN/CNN.h"
#include "../utilities/shop.h"

const float alpha = 0.1;


static void initPolicy(float** policy, int n_states, int n_actions) {
    for (int i = 0; i < n_states; i++) {
        for (int j = 0; j < n_actions; j++) {
            policy[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

static void initWeights(float** weights, int n_layers, int* layer_sizes) {
    for (int i = 0; i < n_layers - 1; i++) {
        for (int j = 0; j < layer_sizes[i]; j++) {
            for (int k = 0; k < layer_sizes[i + 1]; k++) {
                weights[i][j * layer_sizes[i + 1] + k] = ((float)rand() / RAND_MAX) - 0.5;
            }
        }
    }
}

static void initQ(float** q, int n_states, int n_actions) {
    for (int i = 0; i < n_states; i++) {
        for (int j = 0; j < n_actions; j++) {
            q[i][j] = 0.0;
        }
    }
}

static void initValue(float** value, int n_states) {
    for (int i = 0; i < n_states; i++) {
        value[i] = calloc(1, sizeof(float));
    }
}

RL* createRL(int n_inputs, int n_states, int n_actions, int n_outputs, int n_layers, int* layer_sizes, float gamma) {
    RL* rl = (RL*)malloc(sizeof(RL));
    rl->n_inputs = n_inputs;
    rl->n_states = n_states;
    rl->n_actions = n_actions;
    rl->n_outputs = n_outputs;
    rl->n_layers = n_layers;
    rl->layer_sizes = layer_sizes;
    rl->gamma = gamma;

    rl->weights = allocate2DArray(n_layers, layer_sizes);
    rl->q = allocate2DArray(n_states, n_actions);
    rl->reward = allocate2DArray(n_states, n_actions);
    rl->policy = allocate2DArray(n_states, n_actions);
    rl->value = allocate2DArray(n_states, 1);

    initWeights(rl->weights, n_layers, layer_sizes);
    initQ(rl->q, n_states, n_actions);
    initPolicy(rl->policy, n_states, n_actions);
    initValue(rl->value, n_states);

    return rl;
}

void destroyRL(RL* rl) {
    int i;
    for (i = 0; i < rl->n_layers; i++) {
        free(rl->weights[i]);
    }
    free(rl->weights);

    for (i = 0; i < rl->n_states; i++) {
        free(rl->q[i]);
        free(rl->reward[i]);
        free(rl->policy[i]);
    }
    free(rl->q);
    free(rl->reward);
    free(rl->policy);
    free(rl->value);
    free(rl->layer_sizes);
    free(rl);
}

// The Q-learning update rule is Q(s,a) = R(s,a) + γ * max(Q(s',a')) 
// where R is the reward and γ is the discount factor.
void updateRLQ(RL* rl) {
    for (int i = 0; i < rl->n_states; i++) {
        for (int j = 0; j < rl->n_actions; j++) {
            float max_q = FLT_MIN;
            for (int k = 0; k < rl->n_states; k++) {
                max_q = fmax(max_q, rl->q[k][j]);
            }
            rl->q[i][j] = rl->reward[i][j] + rl->gamma * max_q;
        }
    }
}

void updateRLNewQ(RL* rl, float newQ[rl->n_states][rl->n_actions]) {
    for (int i = 0; i < rl->n_states; i++) {
        for (int j = 0; j < rl->n_actions; j++) {
            rl->q[i][j] = newQ[i][j];
        }
    }
}

void updateRLPolicy(RL* rl) {
    for (int i = 0; i < rl->n_states; i++) {
        int best_action = 0;
        for (int j = 1; j < rl->n_actions; j++) {
            if (rl->q[i][j] > rl->q[i][best_action]) {
                best_action = j;
            }
        }
        rl->policy[i] = best_action;
    }
}

void updateRLNewPolicy(RL* rl, float** newPolicy) {
    for (int i = 0; i < rl->n_states; i++) {
        for (int j = 0; j < rl->n_actions; j++) {
            rl->policy[i][j] = newPolicy[i][j];
        }
    }
}

void updateRLValues(RL* rl) {
    for (int i = 0; i < rl->n_states; i++) {
        for (int j = 0; j < rl->n_actions; j++) {
            rl->value[i] += rl->policy[i][j] * rl->q[i][j];
        }
    }
}

void updateRLNewValue(RL* rl, float* newValue) {
    for (int i = 0; i < rl->n_states; i++) {
        rl->value[i] = newValue[i];
    }
}

void trainRLusingCNN(RL* rl, CNN* cnn) {
    int i, j, k, maxIndex;
    float maxValue;
    float* output = (float*)calloc(rl->n_outputs, sizeof(float));

    for (i = 0; i < rl->n_states; i++) {
        for (j = 0; j < rl->n_actions; j++) {
            forwardPropCNN_A(cnn, rl->states[i]);
            rl->q[i][j] = dot(output, rl->weights[j], rl->n_outputs);
        }
        maxValue = -FLT_MAX;
        maxIndex = 0;
        for (j = 0; j < rl->n_actions; j++) {
            if (rl->q[i][j] > maxValue) {
                maxValue = rl->q[i][j];
                maxIndex = j;
            }
        }
        for (j = 0; j < rl->n_actions; j++) {
            rl->policy[i][j] = (j == maxIndex) ? 1 : 0;
        }
    }

    // update values
    updateRLValues(rl);
 
    // normalize policy
    for (i = 0; i < rl->n_states; i++) {
        normalize(rl->policy[i], rl->n_actions);
    }

    // update weights
    for (i = 0; i < rl->n_layers - 1; i++) {
        for (j = 0; j < rl->layer_sizes[i]; j++) {
            for (k = 0; k < rl->layer_sizes[i + 1]; k++) {
                rl->weights[i][j][k] += alpha * gradient[i][j][k];
            }
        }
    }

    // update Q values
    for (i = 0; i < rl->n_states; i++) {
        for (j = 0; j < rl->n_actions; j++) {
            rl->q[i][j] += alpha * (rl->reward[i][j] + rl->gamma * rl->value[i] - rl->q[i][j]);
        }
    }

    // update policy
    for (i = 0; i < rl->n_states; i++) {
        for (j = 0; j < rl->n_actions; j++) {
            rl->policy[i][j] = exp(rl->q[i][j]);
        }
    }
}

float RLpredictCNN(RL* rl, float* cnn_output) {
    int i, j;
    float maxQ = -FLT_MAX;
    int maxQIndex = -1;
 
    for (i = 0; i < rl->n_states; i++) {
        for (j = 0; j < rl->n_actions; j++) {
            if (rl->q[i][j] > maxQ) {
                maxQ = rl->q[i][j];
                maxQIndex = j;
            }
        }
    }

    return maxQIndex;
}

int decideCNNorGAN(RL* rl, float* cnn_data, float** gan_data) {
    // Perform forward propagation on the CNN with the cnn_data
    float* cnn_output = forwardPropCNN_A(rl->cnn, cnn_data);

    // Perform forward propagation on the GAN with the gan_data
    float** gan_output = forwardPropGAN(rl->gan, gan_data);

    // Compare the output of the CNN and GAN, and return 1 if CNN should be used and 0 if GAN should be used
    if (cnn_output > gan_output) {
        return 1;
    }
    else {
        return 0;
    }
}

void selectRLAction(RL* rl, char* buffer) {
    int i, j;
    float max_value = -1.0f;
    int max_index = -1;
    for (i = 0; i < rl->n_states; i++) {
        for (j = 0; j < rl->n_actions; j++) {
            if (rl->q[i][j] > max_value) {
                max_value = rl->q[i][j];
                max_index = j;
            }
        }
    }
    sprintf(buffer, "%d", max_index);
}