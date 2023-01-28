#include "../RL/RL.h"
#include "../GAN/GAN.h"
#include "../CNN/CNN.h"
#include "../Discriminator/Discriminator.h"
#include "../utilities/shop.h"

CARL_Protocol* initCARL_Protocol_Type(CNN* generator, Discriminator* discriminator, GAN* arl_gan, RL* arl_rl, RC_protocol* rc,
                      int ca_batch_size, float generator_learning_rate, float discriminator_learning_rate,
                      int ca_n_inputs, int ca_n_outputs, int ca_n_generator_layers, int ca_n_discriminator_layers,
                      int* ca_generator_layer_sizes, int* ca_discriminator_layer_sizes,
                      int* ca_generator_filter_sizes, int ca_generator_n_filters, 
                      int* ca_discriminator_filter_sizes, int ca_discriminator_n_filters,
                      int ca_generator_batch_size, int ca_discriminator_batch_size,
                      int ca_generator_epochs, int ca_discriminator_epochs,
                      int arl_batch_size, int arl_n_steps,
                      int rc_n_inputs, int rc_n_states, int rc_n_actions, int rc_n_outputs,
                      int rc_n_layers, int* rc_layer_sizes, float rc_gamma, 
                      void (*rc_adjustWeights)(RC_protocol*), void (*rc_adjustGANWeights)(RC_protocol*)) {

    CARL_Protocol* carl = (CARL_Protocol*) malloc(sizeof(CARL_Protocol));
    carl->ca = *initCA_Protocol(generator, discriminator, ca_batch_size, ca_n_inputs, ca_n_outputs);
    carl->arl = *initARL_Protocol(arl_gan, arl_rl, arl_batch_size, arl_n_steps);
    carl->rc = *initRC_protocol(rc_n_inputs, rc_n_states, rc_n_actions, rc_n_outputs, rc_layer_sizes, rc_n_layers, rc_gamma);

    return carl;
}

void initCARL_Protocol_Members(CARL_Protocol* carl, 
                      int ca_batch_size, float generator_learning_rate, float discriminator_learning_rate,
                      int ca_n_inputs, int ca_n_outputs, int ca_n_generator_layers, int ca_n_discriminator_layers,
                      int* ca_generator_layer_sizes, int* ca_discriminator_layer_sizes,
                      int* ca_generator_filter_sizes, int ca_generator_n_filters, 
                      int* ca_discriminator_filter_sizes, int ca_discriminator_n_filters,
                      int ca_generator_batch_size, int ca_discriminator_batch_size,
                      int ca_generator_epochs, int ca_discriminator_epochs,
                      int arl_batch_size, int arl_n_steps,
                      int rc_n_inputs, int rc_n_states, int rc_n_actions, int rc_n_outputs,
                      int rc_n_layers, int* rc_layer_sizes, float rc_gamma, 
                      void (*rc_adjustWeights)(RC_protocol*), void (*rc_adjustGANWeights)(RC_protocol*)) {
    // Initialize the CA_Protocol component
    carl->ca.batch_size = ca_batch_size;
    carl->ca.generator_learning_rate = generator_learning_rate;
    carl->ca.discriminator_learning_rate = discriminator_learning_rate;
    carl->ca.n_inputs = ca_n_inputs;
    carl->ca.n_outputs = ca_n_outputs;
    carl->ca.n_generator_layers = ca_n_generator_layers;
    carl->ca.n_discriminator_layers = ca_n_discriminator_layers;
    carl->ca.generator_layer_sizes = ca_generator_layer_sizes;
    carl->ca.discriminator_layer_sizes = ca_discriminator_layer_sizes;
    carl->ca.generator_filter_sizes = ca_generator_filter_sizes;
    carl->ca.generator_n_filters = ca_generator_n_filters;
    carl->ca.discriminator_filter_sizes = ca_discriminator_filter_sizes;
    carl->ca.discriminator_n_filters = ca_discriminator_n_filters;
    carl->ca.generator_batch_size = ca_generator_batch_size;
    carl->ca.discriminator_batch_size = ca_discriminator_batch_size;
    carl->ca.generator_epochs = ca_generator_epochs;
    carl->ca.discriminator_epochs = ca_discriminator_epochs;

    // Initialize the ARL_Protocol component
        carl->arl.batch_size = arl_batch_size;
    carl->arl.n_steps = arl_n_steps;
    carl->arl.current_step = 0;
    carl->arl.generator_loss = 0;
    carl->arl.discriminator_loss = 0;
    carl->arl.generator_error = (float**)malloc(sizeof(float*));
    carl->arl.discriminator_error = (float**)malloc(sizeof(float*));

    // Initialize the RC Protocol component
    carl->rc.n_inputs = rc_n_inputs;
    carl->rc.n_states = rc_n_states;
    carl->rc.n_actions = rc_n_actions;
    carl->rc.n_outputs = rc_n_outputs;
    carl->rc.weights = (float**)malloc(sizeof(float*));
    carl->rc.n_layers = rc_n_layers;
    carl->rc.layer_sizes = rc_layer_sizes;
    carl->rc.q = (float**)malloc(sizeof(float*));
    carl->rc.reward = (float**)malloc(sizeof(float*));
    carl->rc.policy = (float**)malloc(sizeof(float*));
    carl->rc.value = (float**)malloc(sizeof(float*));
    carl->rc.gamma = rc_gamma;
    carl->rc.adjustWeights = rc_adjustWeights;
    carl->rc.adjustGANWeights = rc_adjustGANWeights;
}

float calculateGaussianDistributionProbability(float input, float mean, float stddev) {
    float exponent = -pow(input - mean, 2) / (2 * pow(stddev, 2));
    return (1 / (sqrt(2 * M_PI) * stddev)) * exp(exponent);
}


void predictNaiveBayes(CARL_Protocol* carl, float** input_data, int n_samples) {
    // Initialize variables to store the probabilities of each output class
    float* class_probs = malloc(sizeof(float) * carl->rc.n_outputs);
    for (int i = 0; i < carl->rc.n_outputs; i++) {
        class_probs[i] = 1.0;
    }

    // Loop through each input sample
    for (int i = 0; i < n_samples; i++) {
        // Loop through each input feature
        for (int j = 0; j < carl->ca.n_inputs; j++) {
            // Loop through each output class
            for (int k = 0; k < carl->rc.n_outputs; k++) {
                // Calculate the probability of the current feature value given the current output class
                float feature_prob = calculateGaussianDistributionProbability(input_data[i][j], carl->rc.input_class_mean[j][k], carl->rc.input_class_stddev[j][k]);
                // Multiply the probability by the current class probability
                class_probs[k] *= feature_prob;
            }
        }

        // Normalize the class probabilities
        float sum = 0.0;
        for (int j = 0; j < carl->rc.n_outputs; j++) {
            sum += class_probs[j];
        }
        for (int j = 0; j < carl->rc.n_outputs; j++) {
            class_probs[j] /= sum;
        }

        // Find the class with the highest probability and set it as the prediction for the current sample
        int max_class = 0;
        for (int j = 1; j < carl->rc.n_outputs; j++) {
            if (class_probs[j] > class_probs[max_class]) {
                max_class = j;
            }
        }
        carl->rc.predictions[i] = max_class;

        // Reset the class probabilities for the next sample
        for (int j = 0; j < carl->rc.n_outputs; j++) {
            class_probs[j] = 1.0;
        }
    }

    free(class_probs);
}

CA_Protocol initCA_Protocol(CNN* generator, Discriminator* discriminator, int batch_size, int n_inputs, int n_outputs) {
    CA_Protocol* protocol = (CA_Protocol*)malloc(sizeof(CA_Protocol));
    protocol->generator = generator;
    protocol->discriminator = discriminator;
    protocol->batch_size = batch_size;
    protocol->n_inputs = n_inputs;
    protocol->n_outputs = n_outputs;
    protocol->n_generator_layers = generator->n_layers;
    protocol->n_discriminator_layers = discriminator->layers;
    protocol->generator_layer_sizes = generator->layer_sizes;
    protocol->discriminator_layer_sizes = discriminator->layer_sizes;
    protocol->generator_filter_sizes = generator->filter_sizes;
    protocol->generator_n_filters = generator->n_filters;
    protocol->discriminator_filter_sizes = discriminator->filter_sizes;
    protocol->discriminator_n_filters = discriminator->n_filters;
    protocol->generator_batch_size = generator->batch_size;
    protocol->discriminator_batch_size = discriminator->batch_size;
    protocol->generator_epochs = generator->epochs;
    protocol->discriminator_epochs = discriminator->epochs;
    protocol->generator_steps = generator->steps;
    protocol->discriminator_steps = discriminator->steps;
    protocol->generator_iterations = 0;
    protocol->discriminator_iterations = 0;
    protocol->generator_batch_iterations = 0;
    protocol->discriminator_batch_iterations = 0;
    protocol->generator_batch_steps = 0;
    protocol->discriminator_batch_steps = 0;
    protocol->generator_learning_rate = generator->learning_rate;
    protocol->discriminator_learning_rate = discriminator->learning_rate;
    protocol->real_data = allocateMatrix(protocol->batch_size, protocol->n_inputs);
    protocol->generator_output = allocateMatrix(protocol->batch_size, protocol->n_outputs);
    protocol->discriminator_output = allocateMatrix(protocol->batch_size, protocol->n_outputs);
    protocol->generator_error = allocateMatrix(protocol->batch_size, protocol->n_outputs);
    protocol->discriminator_error = allocateMatrix(protocol->batch_size, protocol->n_outputs);
    protocol->discriminator_error = allocateMatrix(protocol->batch_size, protocol->n_outputs);
    protocol->generator_learning_rate = 0.01;
    protocol->discriminator_learning_rate = 0.01;
    protocol->n_inputs = n_inputs;
    protocol->n_outputs = n_outputs;
    protocol->n_generator_layers = generator->n_layers;
    protocol->n_discriminator_layers = discriminator->n_layers;
    protocol->generator_layer_sizes = (int*)malloc(sizeof(int) * generator->n_layers);
    protocol->discriminator_layer_sizes = (int*)malloc(sizeof(int) * discriminator->n_layers);
    protocol->generator_weights = allocateMatrix(generator->n_layers, n_inputs);
    protocol->generator_biases = allocateMatrix(generator->n_layers, 1);
    protocol->discriminator_weights = allocateMatrix(discriminator->n_layers, n_inputs);
    protocol->discriminator_biases = allocateMatrix(discriminator->n_layers, 1);
    protocol->generator_filter_sizes = (int*)malloc(sizeof(int) * n_generator_layers);
    protocol->generator_n_filters = generator->n_filters;
    protocol->discriminator_filter_sizes = (int*)malloc(sizeof(int) * discriminator->n_layers);
    protocol->discriminator_n_filters = discriminator->n_filters;
    protocol->generator_batch_size = generator->batch_size;
    protocol->discriminator_batch_size = discriminator->batch_size;
    protocol->generator_epochs = generator->epochs;
    protocol->discriminator_epochs = discriminator->epochs;
    protocol->generator_steps = generator->steps;
    protocol->discriminator_steps = discriminator->steps;
    protocol->generator_iterations = generator->iterations;
    protocol->discriminator_iterations = discriminator->iterations;
    protocol->generator_batch_iterations = generator->batch_iterations;
    protocol->discriminator_batch_iterations = discriminator->batch_iterations;
    protocol->generator_batch_steps = generator->batch_steps;
    protocol->discriminator_batch_steps = discriminator->batch_steps;
    return protocol;
}


ARL_Protocol* initARL_Protocol(ARL_Protocol* protocol, GAN* gan, RL* rl, int batch_size, int n_steps) {
    ARL_Protocol* protocol = (ARL_Protocol*)malloc(sizeof(ARL_Protocol));
    protocol->gan = gan;
    protocol->rl = rl;
    protocol->batch_size = batch_size;
    protocol->n_steps = n_steps;
    protocol->current_step = 0;
    protocol->generator_loss = 0;
    protocol->discriminator_loss = 0;
    protocol->generator_error = allocateMatrix(batch_size, gan->generator->n_outputs);
    protocol->discriminator_error = allocateMatrix(batch_size, gan->discriminator->n_outputs);
    return protocol;
}

RC_Protocol* initRC_protocol(int n_inputs, int n_states, int n_actions, int n_outputs, int* layer_sizes, int n_layers, float gamma) {
    RC_Protocol* rc = (RC_Protocol*)malloc(sizeof(RC_Protocol));
    rc->n_inputs = n_inputs;
    rc->n_states = n_states;
    rc->n_actions = n_actions;
    rc->n_outputs = n_outputs;
    rc->layer_sizes = layer_sizes;
    rc->n_layers = n_layers;
    rc->gamma = gamma;

    // Allocate memory for weights and other arrays
    rc->weights = (float**) malloc(sizeof(float*) * n_layers);
    rc->q = (float**) malloc(sizeof(float*) * n_states);
    rc->reward = (float**) malloc(sizeof(float*) * n_states);
    rc->policy = (float**) malloc(sizeof(float*) * n_states);
    rc->value = (float**) malloc(sizeof(float*) * n_states);

    for (int i = 0; i < n_layers; i++) {
        rc->weights[i] = (float*) malloc(sizeof(float) * layer_sizes[i]);
    }

    for (int i = 0; i < n_states; i++) {
        rc->q[i] = (float*) malloc(sizeof(float) * n_actions);
        rc->reward[i] = (float*) malloc(sizeof(float) * n_actions);
        rc->policy[i] = (float*) malloc(sizeof(float) * n_actions);
        rc->value[i] = (float*) malloc(sizeof(float) * n_actions);
    }

    initCNN(&rc->cnn, n_inputs, n_outputs, n_layers, layer_sizes);
    initRL(&rc->rl, &rc->gan, n_inputs, n_states, n_actions, n_outputs, layer_sizes, gamma);
    initGAN(&rc->gan, &rc->cnn, n_inputs, n_outputs);

    //initialize the functions
    rc->adjustWeights = &adjustWeights;
    rc->adjustGANWeights = &adjustGANWeights;

    return rc;
}


void trainCA_Protocol(CA_Protocol* ca, float** real_data) {
    ca->real_data = real_data;

    // training generator
    float** generator_input = randomNoise(ca->batch_size, ca->n_inputs);
    float** generator_output = forwardPropCNN(ca->generator, generator_input);
    float** discriminator_output = forwardPropCNN(ca->discriminator, generator_output);
    ca->generator_loss = generatorLoss(discriminator_output);
    ca->generator_error = generatorError(discriminator_output);
    backPropCNN(ca->generator, ca->generator_error, ca->generator_learning_rate);

    // training discriminator
    float** real_output = forwardPropCNN(ca->discriminator, real_data);
    float** fake_output = forwardPropCNN(ca->discriminator, generator_output);
    ca->discriminator_loss = discriminatorLoss(real_output, fake_output);
    ca->discriminator_error = discriminatorError(real_output, fake_output);
    backPropCNN(ca->discriminator, ca->discriminator_error, ca->discriminator_learning_rate);
}

// TODO define these functions
void trainARL_Protocol_Data(ARL_Protocol* arl, float** real_data) {
    for (int i = 0; i < arl->n_steps; i++) {
        // Train the GAN
        float** generated_data = generateGANData(arl->gan, arl->batch_size);
        float** discriminator_inputs = concatenateRealAndGeneratedData(real_data, generated_data, arl->batch_size);
        float** discriminator_outputs = evaluateDiscriminator(arl->gan, discriminator_inputs, arl->batch_size);
        float** generator_targets = createGeneratorTargets(arl->batch_size);
        arl->generator_loss = calculateGeneratorLoss(discriminator_outputs, generator_targets, arl->batch_size);
        arl->generator_error = calculateGeneratorError(discriminator_outputs, generator_targets, arl->batch_size);
        adjustGeneratorWeights(arl->gan, arl->generator_error);
        
        // Train the RL
        float** rl_inputs = generated_data;
        float** rl_outputs = evaluateRL(arl->rl, rl_inputs, arl->batch_size);
        float** rl_targets = createRLTargets(arl->batch_size);
        float rl_loss = calculateRLLoss(rl_outputs, rl_targets, arl->batch_size);
        float** rl_error = calculateRLError(rl_outputs, rl_targets, arl->batch_size);
        adjustRLWeights(arl->rl, rl_error);

        arl->current_step++;
    }
}


void trainARL_Protocol_Steps(ARL_Protocol* arl, int n_steps) {
    for (int i = 0; i < n_steps; i++) {
        // Train GAN
        trainGAN(arl->gan, arl->batch_size);
        // Update generator weights and biases in RL
        updateRL(arl->rl, arl->gan->generator_weights, arl->gan->generator_biases);
        // Train RL
        trainRL(arl->rl, arl->batch_size);
        // Update discriminator weights and biases in GAN
        updateGAN(arl->gan, arl->rl->discriminator_weights, arl->rl->discriminator_biases);
    }
}

void trainRLC(RLC_protocol* rlc, float** input, float** expected_output, int n_samples, int n_epochs) {
    for (int i = 0; i < n_epochs; i++) {
        for (int j = 0; j < n_samples; j++) {
            // Forward propagate the input through the CNN
            forwardPropagationCNN(&(rc->cnn), input[j]);
            // Use the output of the CNN as the input for the RL
            rc->rl.n_inputs = rc->cnn.n_outputs;
            // Update the Q-values, rewards, policy and value using the RL algorithm
            updateRLValues(&(rc->rl));
            // Adjust the weights and biases of the CNN using the output of the RL
            rc->adjustWeights(rlc);
            // Adjust the weights and biases of the GAN using the output of the RL
            rc->adjustGANWeights(rlc);
            // Compare the output of the CNN to the expected output and calculate the error
            float error = calculateError(&(rc->cnn), expected_output[j]);
            // Backpropagate the error through the CNN to update the weights and biases
            backpropagationCNN(&(rc->cnn), error);
        }
    }
}



float** generate(CA_Protocol* ca, float** real_data, int n_samples) {
    float** generated_data = malloc(sizeof(float*) * n_samples);
    for (int i = 0; i < n_samples; i++) {
        generated_data[i] = generateSample(ca->generator, real_data[i % ca->batch_size], ca->n_inputs);
    }
    return generated_data;
}

float** evaluate(CNN_GAN_Protocol* protocol, float** generated_data, float** real_data, int n_samples) {
    int n_outputs = protocol->discriminator->n_outputs;
    float** evaluation = (float**) malloc(n_samples * sizeof(float*));

    for (int i = 0; i < n_samples; i++) {
        evaluation[i] = (float*) malloc(n_outputs * sizeof(float));
        evaluation[i] = predict(protocol->discriminator, generated_data[i]);
    }

    return evaluation;
}

float** evaluateCA_Protocol(CA_Protocol* ca, float** real_data) {
    int n_samples = sizeof(real_data) / sizeof(real_data[0]);
    float** generated_data = ca->generate(ca, real_data, n_samples);
    float** evaluated_data = protocol->evaluate(protocol, generated_data, real_data, n_samples);
    return evaluated_data;
}

float** evaluateRC(RLC_protocol* rlc, float** test_input) {
    // 1. Pass the test input through the CNN
    rc->cnn.input = test_input;
    forwardPropagationCNN(&rc->cnn);

    // 2. Pass the CNN output through the GAN
    rc->gan.input = rc->cnn.output;
    forwardPropagationGAN(&rc->gan);

    // 3. Pass the GAN output through the RL
    rc->rl.input = rc->gan.output;
    forwardPropagationRL(&rc->rl);

    // 4. Return the output of the RL
    return rc->rl.output;
}


static void forwardLayer(float* current_input, float** weights, int n_inputs, int layer_size) {
    float* output = (float*) calloc(layer_size, sizeof(float));
    for (int i = 0; i < layer_size; i++) {
        for (int j = 0; j < n_inputs; j++) {
            output[i] += current_input[j] * weights[i][j];
        }
    }
    current_input = output;
}

void forward(RC_Protocol* rc) {
    float* current_input = rc->input_data;
    for (int i = 0; i < rc->n_layers; i++) {
        current_input = forwardLayer(current_input, rc->weights[i], rc->n_inputs, rc->layer_sizes[i]);
    }
    rc->output_data = current_input;
}

void forward(ARL_Protocol* arl, float** input_data) {
    float** q = arl->q;
    float** reward = arl->reward;
    float** policy = arl->policy;
    float** value = arl->value;
    int n_inputs = arl->n_inputs;
    int n_states = arl->n_states;
    int n_actions = arl->n_actions;
    int n_outputs = arl->n_outputs;
    float** weights = arl->weights;
    int n_layers = arl->n_layers;
    int* layer_sizes = arl->layer_sizes;

    //Forward propagation through layers
    float** current_input = input_data;
    for(int i = 0; i < n_layers; i++) {
        forwardLayer(current_input, weights[i], n_inputs, layer_sizes[i]);
        current_input = weights[i];
        n_inputs = layer_sizes[i];
    }

    //Calculate Q-values, rewards, policy, and value from final output
    for(int i = 0; i < n_states; i++) {
        for(int j = 0; j < n_actions; j++) {
            q[i][j] = current_input[i][j];
        }
        reward[i] = current_input[i][n_actions];
        policy[i] = current_input[i][n_actions + 1];
        value[i] = current_input[i][n_actions + 2];
    }
}


void connectRCtoCA(RC_protocol* rc, CA_Protocol* ca) {
    //connect the output of the RC_Protocol to the input of the CA_Protocol
    ca->discriminator->input = rc->output_data;
}


void connectARLtoRC_A(ARL_Protocol* arl, RC_protocol* rc) {
    //connect the output of the ARL_Protocol to the input of the RC_Protocol
    rc->n_inputs = arl->rl->n_states;
    rc->input_data = arl->rl->q;
}

void connectARLtoRC_B(CARL_Protocol* carl) {
    int n_inputs = carl->ca.batch_size * carl->ca.n_outputs;
    int n_outputs = carl->arl.batch_size * carl->arl.n_steps;
    carl->rc.n_inputs = n_inputs;
    carl->rc.n_outputs = n_outputs;
    carl->rc.input = carl->arl.output;
}

void connectARLtoCA(CARL_Protocol* carl, float** ca_output) {
    carl->arl.gan->generator->input = ca_output;
}

void connectCAtoARL(CA_Protocol* ca, ARL_Protocol* arl) {
    //connect the output of the CA_Protocol to the input of the ARL_Protocol
    arl->gan->input = ca->generator->output;
}

void connectCARL(CARL_Protocol* carl) {
    connectCAtoARL(&(carl->ca), &(carl->arl));
    connectARLtoRC(&(carl->arl), &(carl->rc));
    connectRCtoCA(&(carl->rc), &(carl->ca));
}

void backpropCA_Protocol(CARL_Protocol* carl, ARL_Protocol* arl) {
    // Get the generator and discriminator from the CA_Protocol
    CNN* generator = carl->ca.generator;
    Discriminator* discriminator = carl->ca.discriminator;
    
    // Get the current state of the ARL_Protocol
    float** q = arl->q;
    float** reward = arl->reward;
    float** policy = arl->policy;
    float** value = arl->value;
    float gamma = arl->gamma;
    
    // Backpropagate through the generator and discriminator
    backpropCNN(generator, q, reward, policy, value, gamma);
    backpropDiscriminator(discriminator, q, reward, policy, value, gamma);
}


void evaluateCAInputAgainsARLtOutput(CARL_Protocol* carl, float** ca_input, float** arl_output, int n_samples) {
    // Compare the CA_Protocol input against the ARL_Protocol output
    for (int i = 0; i < n_samples; i++) {
        float ca_error = 0;
        float arl_error = 0;
        for (int j = 0; j < carl->ca.n_inputs; j++) {
            ca_error += fabs(ca_input[i][j] - carl->ca.generated_data[i][j]);
            arl_error += fabs(arl_output[i][j] - carl->arl.rl->q[i][j]);
        }
        printf("CA Error for sample %d: %f\n", i, ca_error);
        printf("ARL Error for sample %d: %f\n", i, arl_error);
    }
}


void displayCARLEvolution(CARL_Protocol* carl) {
    printf("Iteration: %d\n", carl->iteration);
    printf("CA Input: %f\n", carl->ca.input);
    printf("ARL Output: %f\n", carl->arl.output);
    printf("RC Input: %f\n", carl->rc.input);
    printf("RC Output: %f\n", carl->rc.output);
    printf("\n");
}

void displayEvolution(CARL_Protocol* carl) {
    for(int i=0; i<n_iterations; i++){
        connectCARL(carl);
        displayCARLEvolution(carl);
    }
}


float evaluateDiscriminatorARL(ARL_Protocol* arl, float* input_data) {
    float* current_input = input_data;
    for (int i = 0; i < arl->n_layers; i++) {
        current_input = forwardLayer(current_input, arl->weights[i], arl->n_inputs, arl->layer_sizes[i]);
    }
    return current_input[0];
}

float evaluateDiscriminatorRC(discriminator, float* input_data) {
    float* current_input = input_data;
    for (int i = 0; i < arl->n_layers; i++) {
        current_input = forwardLayer(current_input, discriminator->weights[i], discriminator->n_inputs, discriminator->layer_sizes[i]);
    }
    return current_input[0];
}

void evaluateGeneratorRC(RC_Protocol* rc, float** generated_data) {
    float discriminator_output = evaluateDiscriminator(rc->gan->discriminator, generated_data);
    rc->gan->generator_score = discriminator_output;
}

void improveGenerator(RC_Protocol* rc) {
    float learning_rate = 0.01; // Adjustable parameter for the learning rate
    for (int i = 0; i < rc->gan->generator->n_layers; i++) {
        for (int j = 0; j < rc->gan->generator->layer_sizes[i]; j++) {
            for (int k = 0; k < rc->gan->generator->n_inputs; k++) {
                rc->gan->generator->weights[i][j][k] += learning_rate * rc->gan->generator_score[i][j] * rc->gan->generator->weights[i][j][k];
            }
        }
    }
}

void adjustCALearningRate(CA_Protocol* ca, float learning_rate) {
    ca->learning_rate = learning_rate;
}

void adjustARLLearningRate(ARL_Protocol* arl, float learning_rate) {
    arl->learning_rate = learning_rate;
}

void adjustRCLearningRate(RC_Protocol* rc, float learning_rate) {
    rc->learning_rate = learning_rate;
}

void adjustLearningRates(CARL_Protocol* carl, float ca_learning_rate, float arl_learning_rate, float rc_learning_rate) {
    adjustCALearningRate(&(carl->ca), ca_learning_rate);
    adjustARLLearningRate(&(carl->arl), arl_learning_rate);
    adjustRCLearningRate(&(carl->rc), rc_learning_rate);
}