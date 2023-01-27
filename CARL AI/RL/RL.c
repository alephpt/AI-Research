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

static void updateRewards(RL* rl, float** generator_weights, float** generator_biases) {
    // Update rewards based on current generator weights and biases
    // This could involve running the generator with various inputs and comparing the outputs to a desired outcome
    // The specific implementation would depend on the specific algorithm being used for the RL
}

static void updatePolicy(RL* rl, float** generator_weights, float** generator_biases) {
    // Update policy based on current rewards and generator weights and biases
    // This could involve updating the probability distribution over actions or updating the value function
    // The specific implementation would depend on the specific algorithm being used for the RL
}

void updateRL(RL* rl, float** generator_weights, float** generator_biases) {
    // update the rewards and policy based on the generator's current weights and biases
    updateRewards(rl, generator_weights, generator_biases);
    updatePolicy(rl, generator_weights, generator_biases);
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

void adjustGANwithRL(RL* rl, GAN* gan) {
    // Perform RL algorithm to adjust the weights and biases of the GAN
    // ...
    // Update the GAN with the new weights and biases
    gan->weights = rl->weights;
    gan->generator->weights = rl->weights;
    gan->generator->biases = rl->weights;
    gan->discriminator->weights = rl->weights;
    gan->discriminator->biases = rl->weights;
}

void adjustGeneratorWeights(CNN* generator, float** rl_weights) {
    for (int i = 0; i < generator->n_layers; i++) {
        for (int j = 0; j < generator->layer_sizes[i]; j++) {
            generator->weights[i][j] += rl_weights[i][j];
        }
    }
}

void adjustDiscriminatorWeights(Discriminator* discriminator, float** rl_weights) {
    int n_weights = discriminator->n_layers * discriminator->n_inputs * discriminator->n_outputs;
    for (int i = 0; i < n_weights; i++) {
        discriminator->weights[i] += rl_weights[i];
    }
}


void updateGANwithRL(RL* rl, GAN* gan) {
    float** q_values = getQValues(rl);
    int action = selectAction(rl, q_values);
    float reward = getReward(rl);

    updateWeights(rl, action, reward, q_values);

    adjustGeneratorWeights(gan->generator, rl->weights);
    adjustDiscriminatorWeights(gan->discriminator, rl->weights);
}

float** getQValues(RL* rl) {
    // Forward pass through the RL network to get the Q-values for all actions
    // Code for forward pass goes here

    return q_values;
}

int selectAction(RL* rl, float** q_values) {
    // Use the epsilon-greedy algorithm or other method to select an action
    // Code for action selection goes here

    return action;
}

float getReward(RL* rl) {
    // Calculate the reward based on the performance of the GAN
    // Code for reward calculation goes here

    return reward;
}

void updateWeights(RL* rl, int action, float reward, float** q_values) {
    // Update the weights of the RL network using the Q-learning algorithm or other method
    // Code for weight update goes here
}

void adjustGeneratorWeights(CNN* generator, float** rl_weights) {
    // Update the weights of the generator network using the weights from the RL network
    // Code for weight update goes here
}

void adjustDiscriminatorWeights(Discriminator* discriminator, float** rl_weights) {
    // Update the weights of the discriminator network using the weights from the RL network
    // Code for weight update goes here
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


void forwardPass(float** weights, float** q, int n_layers, int* layer_sizes, int n_inputs, int batch_size) {
    float** input = q;
    float** output;
    for(int i = 0; i < n_layers; i++) {
        int n_neurons = layer_sizes[i];
        output = allocateMatrix(batch_size, n_neurons);
        for(int j = 0; j < batch_size; j++) {
            for(int k = 0; k < n_neurons; k++) {
                float sum = 0;
                for(int l = 0; l < n_inputs; l++) {
                    sum += input[j][l] * weights[i][l*n_neurons + k];
                }
                output[j][k] = sigmoid(sum);
            }
        }
        n_inputs = n_neurons;
        input = output;
    }
    q = output;
    deallocateMatrix(input, batch_size, n_inputs);
}

float calculateReward(float q) { 
    return q;
}

float calculatePolicy(float reward, float value, float gamma) {
    return reward + gamma * value;
}

float calculateValue(float reward, float value, float gamma) {
    return reward + gamma * value;
}



void updateRewards(RL* rl, float** generator_weights, float** generator_biases) {
    // Update rewards based on generator's performance
    int batch_size = rl->gan->batch_size;
    int n_inputs = rl->gan->n_inputs;
    int n_outputs = rl->gan->n_outputs;
    int n_layers = rl->n_layers;
    int* layer_sizes = rl->layer_sizes;
    float** q = rl->q;
    float** reward = rl->reward;
    float** policy = rl->policy;
    float** value = rl->value;
    float gamma = rl->gamma;

    // Perform forward pass with generator weights and biases
    forwardPass(generator_weights, generator_biases, q, n_layers, layer_sizes, n_inputs, batch_size);

    // Calculate rewards based on generator output
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_outputs; j++) {
            reward[i][j] = calculateReward(q[i][j]);
        }
    }

    // Update policy and value based on rewards
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_outputs; j++) {
            policy[i][j] = calculatePolicy(reward[i][j], value[i][j], gamma);
            value[i][j] = calculateValue(reward[i][j], value[i][j], gamma);
        }
    }
}

void updatePolicy(RL* rl, float** generator_weights, float** generator_biases) {
    for (int i = 0; i < rl->batch_size; i++) {
        for (int j = 0; j < rl->n_outputs; j++) {
            rl->policy[i][j] = calculatePolicy(rl->reward[i][j], rl->value[i][j], rl->gamma);
        }
    }

    // update weights and biases of generator using policy
    updateWeights(rl->gan->generator, generator_weights, generator_biases);
}



void backwardPass(float** weights, float** q, int n_layers, int* layer_sizes, int n_inputs, int batch_size, float gamma) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < layer_sizes[n_layers - 1]; j++) {
            float reward = calculateReward(q[i][j]);
            float value = calculateValue(reward, q[i][j], gamma);
            q[i][j] = calculatePolicy(reward, value, gamma);
        }
    }

    for (int k = n_layers - 2; k >= 0; k--) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < layer_sizes[k]; j++) {
                float sum = 0;
                for (int l = 0; l < layer_sizes[k + 1]; l++) {
                    sum += weights[k][j][l] * q[i][l];
                }
                q[i][j] = sum;
            }
        }
    }

    for (int k = 0; k < n_layers - 1; k++) {
        for (int i = 0; i < layer_sizes[k]; i++) {
            for (int j = 0; j < layer_sizes[k + 1]; j++) {
                float sum = 0;
                for (int l = 0; l < batch_size; l++) {
                    sum += q[l][i] * q[l][j];
                }
                weights[k][i][j] -= sum;
            }
        }
    }
}

void trainRL(RL* rl, int batch_size) {
    // define the training loop
    for (int i = 0; i < rl->gan->generator->epochs; i++) {
        // forward pass
        forwardPass(rl->weights, rl->q, rl->n_layers, rl->layer_sizes, rl->n_inputs, batch_size);
        // update rewards
        updateRewards(rl, rl->gan->generator_weights, rl->gan->generator_biases);
        // update policy
        updatePolicy(rl, rl->gan->generator_weights, rl->gan->generator_biases);
        // backward pass
        backwardPass(rl->weights, rl->q, rl->n_layers, rl->layer_sizes, rl->n_inputs, batch_size, rl->gamma);
    }
}

void evaluateRL(RL* rl, float* state, float* action, float reward, float* next_state) {
    rl->updateQ(rl, state, action, reward, next_state);
    rl->updatePolicy(rl);
}

void adjustRL(RL* rl) {
    // adjust the weights of the RL network
    for (int i = 0; i < rl->n_layers; i++) {
        for (int j = 0; j < rl->layer_sizes[i]; j++) {
            for (int k = 0; k < rl->layer_sizes[i+1]; k++) {
                rl->weights[i][j][k] = updateWeight(rl->weights[i][j][k], rl->q[i][j], rl->reward[i][k], rl->policy[i][j], rl->value[i][k], rl->gamma);
            }
        }
    }
    // adjust the Q, reward, policy, and value matrices
    for (int i = 0; i < rl->n_states; i++) {
        for (int j = 0; j < rl->n_actions; j++) {
            rl->q[i][j] = updateQ(rl->q[i][j], rl->reward[i][j], rl->policy[i][j], rl->value[i][j], rl->gamma);
            rl->reward[i][j] = updateReward(rl->reward[i][j], rl->gan);
            rl->policy[i][j] = updatePolicy(rl->policy[i][j], rl->q[i][j], rl->value[i][j]);
            rl->value[i][j] = updateValue(rl->value[i][j], rl->q[i][j], rl->policy[i][j]);
        }
    }
}

void synchronizeRLwithCNNLayers(RL* rl, CNNLayer* cnn_layers, int n_cnn_layers) {
    // Assign the weights and biases of the CNN layers to the RL
    rl->weights = (float**) malloc(sizeof(float*) * n_cnn_layers);
    for (int i = 0; i < n_cnn_layers; i++) {
        rl->weights[i] = cnn_layers[i].weights;
    }

    // Assign the number of inputs, outputs, and neurons of the CNN layers to the RL
    rl->n_inputs = cnn_layers[0].n_inputs;
    rl->n_outputs = cnn_layers[n_cnn_layers - 1].n_outputs;
    rl->n_layers = n_cnn_layers;
    rl->layer_sizes = (int*) malloc(sizeof(int) * n_cnn_layers);
    for (int i = 0; i < n_cnn_layers; i++) {
        rl->layer_sizes[i] = cnn_layers[i].n_outputs;
    }
}
