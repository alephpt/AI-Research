#ifndef RL_H
#define RL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
typedef struct RL {
    GAN* gan;
    int n_inputs;
    int n_states;
    int n_actions;
    int n_outputs;
    float** weights;
    int n_layers;
    int* layer_sizes;
    float** q;
    float** reward;
    float** policy;
    float** value;
    float gamma;
} RL;

*/


// missing -> real_data
// missing -> discriminator_weights
// missing -> discriminator_biases
typedef struct RL {
    GAN* gan;
    int n_inputs;
    int n_states;       // an integer representing the number of states in the RL problem
    int n_actions;      // an integer representing the number of actions available in the RL problem
    int n_outputs;
    float** weights;
    int n_layers;
    int* layer_sizes;
    float** q;          // a 2D array of floats representing the Q - values for each state - action pair
    float** reward;     // a 2D array of floats representing the rewards for each state - action pair
    float** policy;     // a 2D array of floats representing the policy for each state - action pair
    float** value;      // a 2D array of floats representing the value for each state - action pair
    float gamma;        // a float representing the discount factor for the RL problem
} RL;


RL* createRL(int n_inputs, int n_states, int n_actions, int n_outputs, int n_layers, int* layer_sizes, float gamma);
void updateRL(RL* rl, float** generator_weights, float** generator_biases);
void updateRLQ(RL* rl);
void updateRLNewQ(RL* rl, float newQ[rl->n_states][rl->n_actions]);
void updateRLPolicy(RL* rl);
void updateRLNewPolicy(RL* rl, float** newPolicy);
void updateRLNewValue(RL* rl, float* newValue);
void updateRLValues(RL* rl);
void destroyRL(RL* rl);
void updateRLWeights(RL* rl, float** newWeights);
void saveRL(RL* rl, char* filename);
void loadRL(RL* rl, char* filename);
void receiveRLDataFromCNNandUpdate(RL* rl, char* buffer);
void trainRLusingCNN(RL* rl, CNN* cnn);
int decideCNNorGAN(RL* rl, float* cnn_data, float** gan_data);
float RLpredictCNN(RL* rl, float* cnn_output);
void selectRLAction(RL* rl, char* buffer);

#endif
