#include "../CARL/CARL.h"

// TODO: finish this function
void test_initCA_Protocol() {
    // Initialize generator and discriminator
    CNN* generator = initCNN(...);
    Discriminator* discriminator = initDiscriminator(...);

    // Define batch_size, n_inputs, n_outputs, n_generator_layers, n_discriminator_layers, generator_layer_sizes, discriminator_layer_sizes, generator_filter_sizes, generator_n_filters, discriminator_filter_sizes, discriminator_n_filters, generator_batch_size, discriminator_batch_size, generator_epochs, discriminator_epochs, generator_steps, discriminator_steps, generator_learning_rate, discriminator_learning_rate
    int batch_size = ...;
    int n_inputs = ...;
    int n_outputs = ...;
    int n_generator_layers = ...;
    int n_discriminator_layers = ...;
    int* generator_layer_sizes = ...;
    int* discriminator_layer_sizes = ...;
    int* generator_filter_sizes = ...;
    int generator_n_filters = ...;
    int* discriminator_filter_sizes = ...;
    int discriminator_n_filters = ...;
    int generator_batch_size = ...;
    int discriminator_batch_size = ...;
    int generator_epochs = ...;
    int discriminator_epochs = ...;
    int generator_steps = ...;
    int discriminator_steps = ...;
    float generator_learning_rate = ...;
    float discriminator_learning_rate = ...;

    // Call initCA_Protocol
    CA_Protocol* protocol = initCA_Protocol(generator, discriminator, batch_size, n_inputs, n_outputs, n_generator_layers, n_discriminator_layers, generator_layer_sizes, discriminator_layer_sizes, generator_filter_sizes, generator_n_filters, discriminator_filter_sizes, discriminator_n_filters, generator_batch_size, discriminator_batch_size, generator_epochs, discriminator_epochs, generator_steps, discriminator_steps, generator_learning_rate, discriminator_learning_rate);

    // Assert that the protocol was initialized correctly
    assert(protocol->generator == generator);
    assert(protocol->discriminator == discriminator);
    assert(protocol->batch_size == batch_size);
    assert(protocol->n_inputs == n_inputs);
    assert(protocol->n_outputs == n_outputs);
    assert(protocol->n_generator_layers == n_generator_layers);
    assert(protocol->n_discriminator_layers == n_discriminator_layers);
    assert(protocol->generator_layer_sizes == generator_layer_sizes);
    assert(protocol->discriminator_layer_sizes == discriminator_layer_sizes);
    assert(protocol->generator_filter_sizes == generator_filter_sizes);
    assert(protocol->generator_n_filters == generator_n_filters);
    assert(protocol->discriminator_filter_sizes == discriminator_filter_sizes);
    assert(protocol->discriminator_n_filters == discriminator_n_filters);
    assert(protocol->generator_batch_size == generator_batch_size);
    assert(protocol->discriminator_batch_size == discriminator_batch_size);
    assert(protocol->generator_epochs == generator_epochs);
    assert(protocol->discriminator_epochs == discriminator_epochs);
    assert(protocol->generator_steps == generator_steps);
    assert(protocol->discriminator_steps == discriminator_steps);
    assert(protocol->generator_iterations == 0);
    assert(protocol->discriminator_iterations == 0);
    assert(protocol->generator_batch_iterations == 0);
    assert(protocol->discriminator_batch_iterations == 0);
    assert(protocol->generator_batch_steps == 0);
    assert(protocol->discriminator_batch_steps == 0);
    assert(protocol->generator_learning_rate == generator_learning_rate);
    assert(protocol->discriminator_learning_rate == discriminator_learning_rate);

    freeCA_Protocol(protocol);
}


void test_initARL_Protocol() {
    GAN* gan = initGAN(10, 100, 2, 3, 4, 5, 6, 7, 8, 9);
    RL* rl = initRL(10, 100, 2, 3, 4, 5);
    ARL_Protocol* protocol = initARL_Protocol(gan, rl, 100, 10);

    assert(protocol->gan == gan);
    assert(protocol->rl == rl);
    assert(protocol->batch_size == 100);
    assert(protocol->n_steps == 10);

    freeARL_Protocol(protocol);
    freeGAN(gan);
    freeRL(rl);
}


// TODO: FINISH THIS FUNCTION
void test_protocol_evolution() {
    // Initialize the CNN and Discriminator
    CNN* generator = initCNN(...);
    Discriminator* discriminator = initDiscriminator(...);

    // Initialize the GAN and RL
    GAN* gan = initGAN(generator, discriminator, ...);
    RL* rl = initRL(...);

    // Initialize the CA_Protocol and ARL_Protocol
    CA_Protocol* ca_protocol = initCA_Protocol(generator, discriminator, ...);
    ARL_Protocol* arl_protocol = initARL_Protocol(gan, rl, ...);

    // Train the CA_Protocol and ARL_Protocol for multiple epochs
    for (int i = 0; i < 10; i++) {
        trainCA_Protocol(ca_protocol, ...);
        trainARL_Protocol(arl_protocol, ...);

        // Print the generator and discriminator loss at each epoch
        printf("Epoch %d: Generator Loss = %f, Discriminator Loss = %f\n", i, ca_protocol->generator_loss, 
                ca_protocol->discriminator_loss);
    }

    // Compare the initial generator and discriminator loss to the final loss
    assert(ca_protocol->generator_loss < initial_generator_loss);
    assert(ca_protocol->discriminator_loss < initial_discriminator_loss);

    // Compare the initial RL rewards to the final rewards
    assert(rl->total_rewards > initial_rl_rewards);

    // Clean up
    cleanupCA_Protocol(ca_protocol);
    cleanupARL_Protocol(arl_protocol);
}


void testGANRL(GAN* gan, RL* rl, int n_epochs, int batch_size) {
    // Generate initial fake data
    float** fake_data = generateFakeData(gan->generator, batch_size);
    
    // Train the discriminator on real and fake data
    for (int i = 0; i < n_epochs; i++) {
        trainDiscriminator(gan->discriminator, gan->real_data, fake_data, batch_size);
        float error = calculateDiscriminatorError(gan->discriminator->output, gan->real_data, fake_data);
        
        // Train the generator
        trainGenerator(gan->generator, gan->discriminator, batch_size);
        
        // Update the policy
        updatePolicy(rl, gan->generator->weights, gan->generator->biases);
        
        // Update the generator with the new policy
        adjustGANwithRL(rl, gan);
        
        // Print the current error
        printf("Discriminator error at epoch %d: %f\n", i, error);
    }
}


void test_trainRLC() {
    // Initialize RLC_protocol with appropriate parameters
    RLC_protocol rlc;
    initRLC(&rlc, ...);
    
    // Prepare input and expected output data for training
    float** input = ...;
    float** expected_output = ...;
    int n_samples = ...;
    
    // Train RLC on the input and expected output data for a certain number of epochs
    int n_epochs = ...;
    trainRLC(&rlc, input, expected_output, n_samples, n_epochs);
    
    // Evaluate the trained RLC on a new set of input data
    float** test_input = ...;
    float** test_output = evaluateRLC(&rlc, test_input);
    
    // Compare the test output to the expected output and assert that they are similar
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < rlc.n_outputs; j++) {
            assert(fabs(test_output[i][j] - expected_output[i][j]) < epsilon);
        }
    }
}

void testEvaluateCA_Protocol() {
    // Initialize CNN_GAN_Protocol
    CNN_GAN_Protocol protocol;
    initCNN_GAN_Protocol(&protocol);

    // Generate test input
    float** real_data = generateRealData(protocol.cnn.n_inputs, 10);

    // Evaluate the protocol
    float** generated_data = evaluateCA_Protocol(&protocol, real_data);

    // Assert that the output is not null
    assert(generated_data != NULL);

    // Assert that the number of outputs is correct
    assert(protocol.cnn.n_outputs == getNumberOfColumns(generated_data));

    // Clean up
    free(real_data);
    free(generated_data);
}

void test_trainCA_Protocol() {
    // Initialize test variables
    int n_inputs = 100;
    int n_outputs = 10;
    int n_layers = 2;
    int layer_sizes[] = {128, 64};
    int n_filters = 32;
    int filter_sizes[] = {3, 3};
    int n_samples = 1000;
    int n_epochs = 10;
    float** real_data = generateTestData(n_samples, n_inputs);
    float** expected_output = generateTestData(n_samples, n_outputs);
    
    // Initialize CNN_GAN_Protocol
    CA_Protocol protocol;
    initCA_Protocol(&protocol, n_inputs, n_outputs, n_layers, layer_sizes, n_filters, filter_sizes);
    
    // Train the protocol
    trainCA_Protocol(&protocol, real_data, expected_output, n_samples, n_epochs);
    
    // Evaluate the trained protocol
    float** generated_data = evaluateCA_Protocol(&protocol, real_data);
    
    // Assert that the generated data is similar to the expected output
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_outputs; j++) {
            assert(abs(generated_data[i][j] - expected_output[i][j]) < 0.1);
        }
    }
    
    // Clean up
    free2DArray(real_data, n_samples);
    free2DArray(expected_output, n_samples);
    free2DArray(generated_data, n_samples);
    freeCA_Protocol(&protocol);
}


void test_trainARL_Protocol_Finished() {
    // Create a mock GAN and RL
    GAN gan = createMockGAN();
    RL rl = createMockRL();
    GAN_RL_Protocol protocol = {&gan, &rl, 64, 100, 0, 0, 0, NULL, NULL};

    // Create mock real data
    float** real_data = createMockRealData();

    // Train the protocol
    trainARL_Protocol(&protocol, real_data);

    // Assert that the GAN and RL were trained correctly
    assert(gan.trained == true);
    assert(rl.trained == true);

    // Assert that the generator and discriminator losses were updated
    assert(protocol.generator_loss > 0);
    assert(protocol.discriminator_loss > 0);

    // Assert that the generator and discriminator errors were updated
    assert(protocol.generator_error != NULL);
    assert(protocol.discriminator_error != NULL);

    // Clean up
    freeMockGAN(gan);
    freeMockRL(rl);
    freeMockRealData(real_data);
}

void test_trainARL_Protocol_LossAndError() {
    // Initialize GAN_RL_Protocol with appropriate parameters
    GAN_RL_Protocol protocol;
    protocol.gan = createGAN(...);
    protocol.rl = createRL(...);
    protocol.batch_size = 64;
    protocol.n_steps = 1000;
    protocol.current_step = 0;

    // Generate sample data for testing
    float** real_data = generateSampleData(...);

    // Train the GAN_RL_Protocol on the sample data
    trainGAN_RL_Protocol(&protocol, real_data);

    // Assert that the generator and discriminator losses have decreased
    assert(protocol.generator_loss < initial_generator_loss);
    assert(protocol.discriminator_loss < initial_discriminator_loss);

    // Assert that the generator and discriminator errors have decreased
    assert(protocol.generator_error < initial_generator_error);
    assert(protocol.discriminator_error < initial_discriminator_error);

    // Assert that the current_step has incremented to the appropriate value
    assert(protocol.current_step == protocol.n_steps);

    // Clean up memory
    freeGAN(protocol.gan);
    freeRL(protocol.rl);
    freeSampleData(real_data);
}

void testCARL() {
    int batch_size = 10;
    int n_inputs = 100;
    int n_outputs = 10;
    int n_states = 15;
    int n_actions = 5;
    int n_steps = 100;
    float gamma = 0.9;
    int layer_sizes[] = {32, 64, 128};
    int n_layers = 3;
    
    CNN* generator = initCNN(n_inputs, layer_sizes, n_layers);
    Discriminator* discriminator = initDiscriminator(n_inputs, layer_sizes, n_layers);
    GAN* gan = initGAN(generator, discriminator);
    RL* rl = initRL(n_states, n_actions, n_outputs, layer_sizes, n_layers, gamma);
    float** real_data = createData(batch_size, n_inputs);
    float** generated_data = generateData(generator, real_data, batch_size);
    float** input_data = evaluateData(discriminator, generated_data, real_data, batch_size);
    float** output_data = updateRL(rl, input_data, n_steps);
    float** final_output_data = updateCA(generator, discriminator, output_data, n_steps);
    for (int i = 0; i < n_steps; i++) {
        printf("Step %d:\n", i);
        printf("Real data: ");
        printData(real_data[i], n_inputs);
        printf("Generated data: ");
        printData(generated_data[i], n_inputs);
        printf("Input data to RL: ");
        printData(input_data[i], n_inputs);
        printf("Output data from RL: ");
        printData(output_data[i], n_outputs);
        printf("Final output data: ");
        printData(final_output_data[i], n_outputs);
    }
    freeData(real_data, batch_size);
    freeData(generated_data, batch_size);
    freeData(input_data, batch_size);
    freeData(output_data, batch_size);
    freeData(final_output_data, batch_size);
    freeCNN(generator);
    freeDiscriminator(discriminator);
    freeGAN(gan);
    freeRL(rl);
}


void testCAtoARL() {
    //initialize CA_Protocol and ARL_Protocol
    CA_Protocol ca = initCA_Protocol(/* ... */);
    ARL_Protocol arl = initARL_Protocol(/* ... */);

    //connect CA_Protocol output to ARL_Protocol input
    connectCAtoARL(&ca, &arl);

    //generate some data from the CA_Protocol
    float** ca_output = generate(ca, /* ... */);

    //check that the ARL_Protocol input is the same as the CA_Protocol output
    assert(ca_output == arl.input);

    printf("CA to ARL connection test passed\n");
}

void test_CA_ARL_Connection() {
    // Initialize CA_Protocol and ARL_Protocol
    CNN* generator = initCNN(n_inputs, n_outputs, n_layers, layer_sizes);
    Discriminator* discriminator = initDiscriminator(n_inputs, n_outputs, n_layers, layer_sizes);
    int batch_size = 10;
    CA_Protocol* ca = initCA_Protocol(generator, discriminator, batch_size, n_inputs, n_outputs);
    GAN* gan = initGAN(generator, discriminator);
    RL* rl = initRL(n_inputs, n_states, n_actions, n_outputs, layer_sizes, n_layers, gamma);
    int n_steps = 20;
    ARL_Protocol* arl = initARL_Protocol(gan, rl, batch_size, n_steps);

    // Connect CA_Protocol to ARL_Protocol
    connectCAtoARL(ca, arl);

    // Test connection by running forward pass on CA_Protocol and checking input to ARL_Protocol
    float** ca_output = generate(ca, real_data, n_samples);
    assert(ca_output == arl->rl->input);

    printf("CA_ARL connection test passed\n");
}

void testCAtoARLConnection(CARL_Protocol* carl) {
    // Initialize input data for CA_Protocol
    float** ca_input = createInputData();
    // Run CA_Protocol
    float** ca_output = carl->ca.generate(carl->ca.generator, ca_input, ca_input_size);
    // Connect CA_Protocol output to ARL_Protocol input
    connectCAtoARL(&(carl->ca), &(carl->arl));
    // Run ARL_Protocol
    float** arl_output = carl->arl.gan->generate(carl->arl.gan, carl->arl.rl, carl->arl.batch_size, carl->arl.n_steps);
    // Compare CA_Protocol input to ARL_Protocol output
    for(int i = 0; i < ca_input_size; i++) {
        for(int j = 0; j < carl->ca.n_inputs; j++) {
            if(fabs(ca_input[i][j] - arl_output[i][j]) > 0.1) {
                printf("Error: CA_Protocol input and ARL_Protocol output do not match\n");
                return;
            }
        }
    }
    printf("Test passed: CA_Protocol input and ARL_Protocol output match\n");
}

void functionalTestRLandCA(RL_Protocol* rl, CA_Protocol* ca) {
    // Connect the output of the RL_Protocol to the input of the CA_Protocol
    connectRCtoCA(&(rl->rc), ca);

    // Initialize variables to store input data and output data
    float** input_data = generateRandomData(rl->n_inputs, rl->n_states);
    float** output_data = (float**) malloc(sizeof(float*) * rl->n_states);
    for (int i = 0; i < rl->n_states; i++) {
        output_data[i] = (float*) malloc(sizeof(float) * rl->n_outputs);
    }

    // Run the CA_Protocol on the input data
    ca->forward(ca, input_data, output_data, rl->n_states);

    // Compare the output of the RL_Protocol to the output of the CA_Protocol
    int correct_predictions = 0;
    for (int i = 0; i < rl->n_states; i++) {
        if (fabs(rl->rc.output[i] - output_data[i]) < 0.001) {
            correct_predictions++;
        }
    }

    // Print the accuracy of the comparison
    printf("Accuracy: %f\n", (float) correct_predictions / (float) rl->n_states);

    // Free allocated memory
    for (int i = 0; i < rl->n_states; i++) {
        free(output_data[i]);
    }
    free(output_data);
    for (int i = 0; i < rl->n_inputs; i++) {
        free(input_data[i]);
    }
    free(input_data);
}


void test_connectRCtoCA() {
    RC_protocol rc;
    CA_Protocol ca;

    //Initialize and set values for rc and ca
    rc.n_inputs = 10;
    rc.n_outputs = 5;
    rc.input_data = (float**) malloc(sizeof(float*) * rc.n_inputs);
    rc.output_data = (float**) malloc(sizeof(float*) * rc.n_outputs);
    for (int i = 0; i < rc.n_inputs; i++) {
        rc.input_data[i] = (float*) malloc(sizeof(float));
        rc.input_data[i][0] = i;
    }
    for (int i = 0; i < rc.n_outputs; i++) {
        rc.output_data[i] = (float*) malloc(sizeof(float));
        rc.output_data[i][0] = i * 2;
    }
    ca.n_inputs = rc.n_outputs;
    ca.input_data = (float**) malloc(sizeof(float*) * ca.n_inputs);
    for (int i = 0; i < ca.n_inputs; i++) {
        ca.input_data[i] = (float*) malloc(sizeof(float));
    }

    //connect the rc to the ca
    connectRCtoCA(&rc, &ca);

    //print input and output data of rc and ca to compare
    printf("RC input data: \n");
    for (int i = 0; i < rc.n_inputs; i++) {
        printf("%.2f ", rc.input_data[i][0]);
    }
    printf("\nRC output data: \n");
    for (int i = 0; i < rc.n_outputs; i++) {
        printf("%.2f ", rc.output_data[i][0]);
    }
    printf("\nCA input data: \n");
    for (int i = 0; i < ca.n_inputs; i++) {
        printf("%.2f ", ca.input_data[i][0]);
    }
    printf("\n");

    //deallocate memory
    for (int i = 0; i < rc.n_inputs; i++) {
        free(rc.input_data[i]);
    }
    free(rc.input_data);
    for (int i = 0; i < rc.n_outputs; i++) {
        free(rc.output_data[i]);
    }
    free(rc.output_data);
    for (int i = 0; i < ca.n_inputs; i++) {
        free(ca.input_data[i]);
    }
    free(ca.input_data);
}


void functionalTest_connectARLtoRC(CARL_Protocol* carl) {
    // Prepare test data
    float** input_data = generateTestData(carl->ca.n_inputs, carl->ca.batch_size);
    float** output_data = generateTestData(carl->arl.n_outputs, carl->arl.batch_size);

    // Connect ARL to RC_B
    connectARLtoRC(carl);

    // Feed test data into ARL
    carl->arl.forward(carl->arl, input_data);

    // Compare output of ARL to input of RC_B
    for (int i = 0; i < carl->arl.batch_size; i++) {
        for (int j = 0; j < carl->arl.n_outputs; j++) {
            assert(fabs(carl->arl.output[i][j] - carl->rc.input_data[i][j]) < 1e-5);
        }
    }

    // Clean up
    freeTestData(input_data, carl->ca.n_inputs, carl->ca.batch_size);
    freeTestData(output_data, carl->arl.n_outputs, carl->arl.batch_size);
}

void test_connectARLtoRC(CARL_Protocol* carl) {
    // Prepare input data for the ARL_Protocol
    float** input_data = generateTestData();
    
    // Run the forward function of the ARL_Protocol
    carl->arl.forward(carl->arl, input_data);
    
    // Connect the ARL_Protocol output to the RC_Protocol input
    connectARLtoRC_B(carl);
    
    // Run the forward function of the RC_Protocol
    carl->rc.forward(carl->rc);
    
    // Compare the ARL_Protocol input data to the RC_Protocol output data
    for (int i = 0; i < carl->arl.n_inputs; i++) {
        for (int j = 0; j < carl->arl.n_outputs; j++) {
            assert(fabs(input_data[i][j] - carl->rc.output[i][j]) < 0.01);
        }
    }
    printf("connectARLtoRC_B test passed\n");
}

void test_connectCARL() {
    // Initialize and set values for CARL_Protocol, CA_Protocol, ARL_Protocol, and RC_Protocol
    CARL_Protocol carl;
    CA_Protocol ca;
    ARL_Protocol arl;
    RC_Protocol rc;
    // set values for ca, arl, and rc
    // Connect the protocols
    connectCARL(&carl);
    // Set input data for CA_Protocol
    float** ca_input_data = generate_input_data();
    // Pass input data through CA_Protocol
    ca.forward(ca, ca_input_data);
    // Set expected output data for CA_Protocol
    float** expected_ca_output = generate_expected_ca_output();
    // Compare the output of the CA_Protocol to the expected output
    assert_equal(ca.output, expected_ca_output, "CA_Protocol output does not match expected output");
    // Set input data for ARL_Protocol
    float** arl_input_data = generate_input_data();
    // Pass input data through ARL_Protocol
    arl.forward(arl, arl_input_data);
    // Set expected output data for ARL_Protocol
    float** expected_arl_output = generate_expected_arl_output();
    // Compare the output of the ARL_Protocol to the expected output
    assert_equal(arl.output, expected_arl_output, "ARL_Protocol output does not match expected output");
    // Set input data for RC_Protocol
    float** rc_input_data = generate_input_data();
    // Pass input data through RC_Protocol
    rc.forward(rc, rc_input_data);
    // Set expected output data for RC_Protocol
    float** expected_rc_output = generate_expected_rc_output();
    // Compare the output of the RC_Protocol to the expected output
    assert_equal(rc.output, expected_rc_output, "RC_Protocol output does not match expected output");
    // Output success message
    printf("All protocols connected and tested successfully\n");
}