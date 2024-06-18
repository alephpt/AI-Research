#include "gan.h"
#include <assert.h>

void test_createGAN() {
    int n_inputs = 100;
    int n_outputs = 10;
    int n_layers = 2;
    int layer_sizes[] = { 128, 64 };
    GAN* gan = createGAN(n_inputs, n_outputs, n_layers, layer_sizes);
    assert(gan->n_inputs == n_inputs);
    assert(gan->n_outputs == n_outputs);
    assert(gan->n_layers == n_layers);
    for (int i = 0; i < n_layers; i++) {
        assert(gan->layer_sizes[i] == layer_sizes[i]);
    }
    destroyGAN(gan);
}

void test_updateWeights() {
    int n_inputs = 100;
    int n_outputs = 10;
    int n_layers = 2;
    int layer_sizes[] = { 128, 64 };
    GAN* gan = createGAN(n_inputs, n_outputs, n_layers, layer_sizes);
    float newWeights[n_layers][layer_sizes[0]][layer_sizes[1]] = {
        {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}},
        {{0.7, 0.8, 0.9}, {1.0, 1.1, 1.2}}
    };
    updateWeights(gan, (float**)newWeights);
    for (int i = 0; i < n_layers; i++) {
        for (int j = 0; j < gan->layer_sizes[i]; j++) {
            for (int k = 0; k < gan->layer_sizes[i + 1]; k++) {
                assert(gan->weights[i][j][k] == newWeights[i][j][k]);
            }
        }
    }
    destroyGAN(gan);
}

int main() {
    test_createGAN();
    test_updateWeights();
    printf("All tests passed!");
    return 0;
}