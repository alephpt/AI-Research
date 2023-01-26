#include "cnn.h"
#include <assert.h>

void test_createCNN() {
    int n_inputs = 100;
    int n_outputs = 10;
    int n_layers = 2;
    int filter_sizes[] = { 3, 2 };
    int n_filters[] = { 32, 64 };
    CNN* cnn = createCNN(n_inputs, n_outputs, n_layers, filter_sizes, n_filters);
    assert(cnn->n_inputs == n_inputs);
    assert(cnn->n_outputs == n_outputs);
    assert(cnn->n_layers == n_layers);
    for (int i = 0; i < n_layers; i++) {
        assert(cnn->filter_sizes[i] == filter_sizes[i]);
        assert(cnn->n_filters[i] == n_filters[i]);
    }
    destroyCNN(cnn);
}

void test_updateWeights() {
    int n_inputs = 100;
    int n_outputs = 10;
    int n_layers = 2;
    int filter_sizes[] = { 3, 2 };
    int n_filters[] = { 32, 64 };
    CNN* cnn = createCNN(n_inputs, n_outputs, n_layers, filter_sizes, n_filters);
    float newWeights[n_layers][n_filters[0]][filter_sizes[0]][filter_sizes[0]] = {
        {{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}, {{0.7, 0.8, 0.9}, {1.0, 1.1, 1.2}}},
        {{{0.3, 0.2, 0.1}, {0.6, 0.5, 0.4}, {0.9, 0.8, 0.7}}, {{1.2, 1.1, 1.0}, {1.5, 1.4, 1.3}, {1.8, 1.7, 1.6}}}
    };
    updateWeights(cnn, (float***)newWeights);
    // Assert that the weights have been updated correctly
    for (int i = 0; i < n_layers; i++) {
        for (int j = 0; j < n_filters[i]; j++) {
            for (int k = 0; k < filter_sizes[i]; k++) {
                for (int l = 0; l < filter_sizes[i]; l++) {
                    assert(cnn->weights[i][j][k][l] == newWeights[i][j][k][l]);
                }
            }
        }
    }
    destroyCNN(cnn);
}

int main() {
    test_createCNN();
    test_updateWeights();
    return 0;
}