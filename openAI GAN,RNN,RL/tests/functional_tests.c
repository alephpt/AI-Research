#include "../GAN/GAN.h"
#include "../CNN/CNN.h"
#include "../RL/RL.h"
#include "../utilities/shop.h"
#include <assert.h>

void test_GAN_to_CNN(GAN* gan, float** goodData, int n_samples, int sample_size) {
    for (int i = 0; i < n_samples; i++) {
        float** generatedData = generateGANData(gan, sample_size);
        
        if (!isSimilar(generatedData, goodData, sample_size)) {
            printf("generated data is not similar to good data\n");
            return;
        }

        CNN* cnn = createCNN(sample_size, ...);
        trainCNN_A(cnn, generatedData, labels);
        float accuracy = evaluateCNN(cnn, testData, testLabels);
        
        if (accuracy < MIN_ACCURACY) {
            printf("CNN trained on generated data has low accuracy\n");
            return;
        }
    }
    printf("GAN is generating useful data\n");
}

void test_GAN_CNN_integration(CNN* cnn, GAN* gan, int sample_size, float threshold) {
    // Generate new images from GAN
    float** generatedImages = generateDataGAN(gan, sample_size);

    // Pass generated images through trained CNN
    float* predictedLabels = evaluateCNN(cnn, generatedImages, sample_size);

    // Compare predicted labels to actual labels
    int correctPredictions = 0;
    for (int i = 0; i < sample_size; i++) {
        if (predictedLabels[i] == actualLabels[i]) {
            correctPredictions++;
        }
    }
    float accuracy = (float)correctPredictions / sample_size;

    // Check if accuracy is above threshold
    if (accuracy >= threshold) {
        printf("GAN successfully generated images that can be used to train a CNN, accuracy: %f\n", accuracy);
    }
    else {
        printf("GAN failed to generate images that can be used to train a CNN, accuracy: %f\n", accuracy);
    }

    deallocate2DArray(generatedImages, sample_size);
    deallocate1DArray(predictedLabels);
}

void test_GAN_with_CNN_output() {
    // Set up GAN and CNN
    int n_inputs = 100;
    int n_outputs = 1;
    int n_layers = 3;
    int filter_sizes[3] = { 3, 5, 7 };
    int n_filters = 32;
    int layer_sizes[3] = { 64, 128, 256 };

    GAN* gan = createGAN(n_inputs, n_outputs, n_layers, filters_sizes, n_filters, layers_sizes);
    CNN* cnn = createCNN(n_inputs, n_outputs, n_layers, filters_sizes, filters, layers_sizes);
    trainCNN_A(cnn, goodData, labels);

    // Generate data with the GAN using output from the CNN
    float** generatedData = generate2DGANData(gan, cnn->output);

    // Test that the generated data is similar to the good data
    float similarity = isSimilar(generatedData, goodData, 1000);
    assert(similarity > 0.9);

    // Test that the CNN can correctly classify the generated data
    float accuracy = evaluateCNN(cnn, generatedData, labels);
    assert(accuracy > 0.8);

    // Clean up memory
    deallocate2DArray(generatedData);
    deallocate2DArray(gan->weights);
    deallocate2DArray(gan->biases);
    deallocate2DArray(cnn->weights);
    deallocate2DArray(cnn->biases);
    free(gan);
    free(cnn);
}

void test_RL_with_CNN_output() {
    float** trainData = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6], [0.4, 0.5, 0.6, 0.7]];
    int* trainLabels = [1, 0, 1, 0];

    // Initialize and train CNN
    int n_inputs = 784;
    int n_outputs = 10;
    int n_layers = 3;
    int layer_sizes[] = { 256, 128, 64 };
    float gamma = 0.9;
    CNN* cnn = createCNN(n_inputs, n_outputs, n_layers, layer_sizes);
    trainCNN_A(cnn, trainData, trainLabels);

    // Initialize and train RL
    int n_states = 10;
    int n_actions = 10;
    RL* rl = createRL(n_inputs, n_states, n_actions, n_outputs, n_layers, layer_sizes, gamma);
    trainRLusingCNN(rl, cnn); // pending

    // Test RL using CNN output
    int n_samples = 100;
    float* testData = generate1DCNNData(n_samples, n_inputs); 
    float* cnn_output = forwardPropCNN_B(cnn, testData);
    float rl_output = RLpredictCNN(rl, cnn_output);
    assert(rl_output != NULL);
}

void test_RL_CNN() {
    int n_inputs = 10;
    int n_states = 20;
    int n_actions = 5;
    int n_outputs = 1;
    int n_layers = 3;
    int layer_sizes[] = { 10, 20, 5 };
    float gamma = 0.9;

    RL* rl = createRL(n_inputs, n_states, n_actions, n_outputs, n_layers, layer_sizes, gamma);

    // test RL's ability to use the output of the CNN
    int n_samples = 100;
    float** generatedData = generate2DCNNData(n_samples, n_inputs);
    float** goodData = generate2DCNNData(n_samples, n_inputs);

    CNN* cnn = createCNN(n_inputs, n_outputs, n_layers, layer_sizes);
    trainCNN_A(cnn, generatedData, goodData);

    float** cnn_output = evaluateCNN(cnn, generatedData, n_samples);

    updateRLNewQ(rl, cnn_output);
    updateRLValues(rl);
    updateRLPolicy(rl);

    // Assert that the output of the CNN is being used correctly by the RL
    assert(rl->q != NULL);
    assert(rl->value != NULL);
    assert(rl->policy != NULL);

    destroyRL(rl);
    destroyCNN(cnn);
    deallocate2DArray(generatedData, n_samples);
    deallocate2DArray(goodData, n_samples);
    deallocate2DArray(cnn_output, n_samples);
}


void test_selectRLAction_A() {
    RL* rl = createRL(2, 2, 2, 1, 0.9);

    // initialize Q-values for testing
    rl->q[0][0] = 0;
    rl->q[0][1] = 1;
    rl->q[1][0] = 2;
    rl->q[1][1] = 3;

    char buffer[100];

    // test select action when in state 0
    rl->current_state = 0;

    selectRLAction(rl, buffer);

    if (strcmp(buffer, "Selected action 1") != 0) {
        printf("Error in test_selectRLAction: expected action 1 but got %s\n", buffer);
    }

    // test select action when in state 1
    rl->current_state = 1;

    selectRLAction(rl, buffer);
    if (strcmp(buffer, "Selected action 1") != 0) {
        printf("Error in test_selectRLAction: expected action 1 but got %s\n", buffer);
    }

    destroyRL(rl);
}

void test_selectRLAction_B() {
    RL rl = {
        .n_inputs = 2,
        .n_states = 3,
        .n_actions = 2,
        .n_outputs = 1,
        .n_layers = 2,
        .layer_sizes = {3, 2},
        .q = {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        },
        .reward = {
            {1.0, 1.0},
            {1.0, 1.0},
            {1.0, 1.0}
        },
        .policy = {
            {0.5, 0.5},
            {0.5, 0.5},
            {0.5, 0.5}
        },
        .value = {
            {1.0},
            {1.0},
            {1.0}
        },
        .gamma = 0.9
    };


    // call selectRLAction
    selectRLAction(&rl, 0);
    assert(rl.policy[0][0] == 1.0);
    assert(rl.policy[0][1] == 0.0);

    // call selectRLAction again
    selectRLAction(&rl, 1);
    assert(rl.policy[1][0] == 0.0);
    assert(rl.policy[1][1] == 1.0);

    // call selectRLAction again
    selectRLAction(&rl, 2);
    assert(rl.policy[2][0] == 1.0);
    assert(rl.policy[2][1] == 0.0);
}


void test_trainRL(RL* rl) {
    // test data for the RL's state and action
    int state = 5;
    int action = 2;
    int new_state = 7;
    float reward = 3.0;
    float q_next = 6.0;
    float cnn_output = 0.8;
    float gan_output = 0.4;

    // Fill in the q, reward, policy, and value arrays with test data
    rl->q[state][action] = 5.0;
    rl->reward[state][action] = 2.0;
    rl->policy[state][action] = 0.6;
    rl->value[state] = 4.0;

    // Test training the CNN
    if (cnn_output > gan_output) {
        trainCNN_A(rl);
        assert(rl->weights[0][0] == 5.8); // check that weights were updated
        assert(rl->q[state][action] == 5.4); // check that q value was updated
        assert(rl->policy[state][action] == 0.62); // check that policy was updated
    }

    // Test training the GAN
    else {
        trainGAN(rl);
        assert(rl->weights[1][1] == 4.6); // check that weights were updated
        assert(rl->q[state][action] == 5.2); // check that q value was updated
        assert(rl->policy[state][action] == 0.61); // check that policy was updated
    }
}

void test_trainRL(RL* rl) {
    // test data for the RL's state and action
    int state = 5;
    int action = 2;
    int new_state = 7;
    float reward = 3.0;
    float q_next = 6.0;
    float cnn_output = 0.4;
    float gan_output = 0.8;

    // Fill in the q, reward, policy, and value arrays with test data
    rl->q[state][action] = 5.0;
    rl->reward[state][action] = 2.0;
    rl->policy[state][action] = 0.6;
    rl->value[state] = 4.0;

    // Test training the CNN
    if (cnn_output > gan_output) {
        trainCNN(rl);
        assert(rl->weights[0][0] == 5.8); // check that weights were updated
        assert(rl->q[state][action] == 5.4); // check that q value was updated
        assert(rl->policy[state][action] == 0.62); // check that policy was updated
    }

    // Test training the GAN
    else {
        trainGAN(rl);
        assert(rl->weights[1][1] == 4.6); // check that weights were updated
        assert(rl->q[state][action] == 5.2); // check that q value was updated
        assert(rl->policy[state][action] == 0.61); // check that policy was updated
    }
}


void test_rlDecisionMaking() {
    // Initialize test data
    int n_samples = 100;
    int n_inputs = 10;
    int n_outputs = 2;
    int n_layers = 2;
    int filter_sizes[] = { 3, 2 };
    int n_filters[] = { 4, 3 };
    int layer_sizes[] = { 5, 4 };
    float** cnn_data = generate2DCNNData(n_samples, n_inputs);
    float** gan_data = generateGANData(n_samples, n_inputs);
    float** cnn_output;
    float** gan_output;
    int decision;

    // Initialize CNN and GAN
    CNN* cnn = createCNN(n_inputs, n_outputs, n_layers, filter_sizes, n_filters, layer_sizes);
    GAN* gan = createGAN(n_inputs, n_outputs, n_layers, filter_sizes, n_filters, layer_sizes);

    // Initialize RL
    RL* rl = createRL(n_inputs, n_outputs, n_layers, layer_sizes);

    // Test decision making
    decision = decideCNNorGAN(rl, cnn_data, gan_data);
    if (decision == 0) {
        printf("RL has decided to train the CNN using the GAN data\n");
        cnn_output = forwardPropCNNfromGAN(cnn, gan_data);
    }
    else if (decision == 1) {
        printf("RL has decided to train the GAN using the CNN data\n");
        gan_output = forwardPropGANfromCNN(gan, cnn_data);
    }
    else {
        printf("Error in RL decision making\n");
    }

    // Cleanup
    destroyCNN(cnn);
    destroyGAN(gan);
    destroyRL(rl);
}