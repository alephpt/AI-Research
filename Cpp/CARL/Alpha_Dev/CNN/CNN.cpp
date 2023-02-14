#include "CNN.h"

CNN::CNN() : n_inputs(0), n_input_layers(0) { appendNewLayer(createNewLayer(CNN_CONVOLUTION_LAYER)); }

CNN::CNN(ftensor3d input, int n_inputs)
{
    appendNewLayer(createNewLayer(CNN_CONVOLUTION_LAYER));
    n_input_layers = n_inputs;
    inputs = input;
    n_inputs = (int)inputs.size() / n_input_layers;
}

CNN::~CNN() {
    for (int l = 0; l < n_layers; l++) {
        layers[l]->kernels.clear();
    }

    layers.clear();
    inputs.clear();
    labels.clear();
}

void CNN::addNewLayer(CNNLayerType lt) { appendNewLayer(createNewLayer(lt)); }
void CNN::addNewKernel(FilterStyle fs) { appendNewKernel(createNewKernel(fs)); };
void CNN::addNewKernel(FixedFilterDimensions fd) { appendNewKernel(createNewKernel(fd)); };
void CNN::addNewKernel(FixedFilterDimensions fd, FilterStyle fs) { appendNewKernel(createNewKernel(fd, fs)); };
void CNN::addNewKernel(DynamicFilterDimensions fd, int n) { appendNewKernel(createNewKernel(fd, n)); };
void CNN::addNewKernel(DynamicFilterDimensions fd, int n, FilterStyle fs) { appendNewKernel(createNewKernel(fd, n, fs)); };

int CNN::getLayerCount() { return n_layers; }
CNNLayer* CNN::getCurrentLayer() { return layers[n_layers - 1]; }
CNNLayer* CNN::getPreviousLayer() { return layers[n_layers - 2]; }
void CNN::appendNewKernel(Kernel* kernel) { getCurrentLayer()->kernels.push_back(kernel); getCurrentLayer()->n_kernels++; }
Kernel* CNN::createNewKernel() { return new Kernel(); }
Kernel* CNN::createNewKernel(FilterStyle fs) { return new Kernel(fs); }
Kernel* CNN::createNewKernel(FixedFilterDimensions fd) { return new Kernel(fd); }
Kernel* CNN::createNewKernel(FixedFilterDimensions fd, FilterStyle fs) { return new Kernel(fd, fs); }
Kernel* CNN::createNewKernel(DynamicFilterDimensions fd, int n) { return new Kernel(fd, n); }
Kernel* CNN::createNewKernel(DynamicFilterDimensions fd, int n, FilterStyle fs) { return new Kernel(fd, n, fs); }
void CNN::appendNewLayer(CNNLayer* layer) { layers.push_back(layer); n_layers++; }


CNNLayer* CNN::createNewLayer(CNNLayerType layer_type) {
    CNNLayer* new_layer = new CNNLayer;

    new_layer->layer_type = layer_type;

    return new_layer;
}

void CNN::printCNN() {
    printf("Convolutional Neural Network: \n");
    printf("Number of Inputs: \t\t%d\n", n_inputs);
    printf("Number of Layers per Input: \t%d\n", n_input_layers);
    printf("Number of CNN Layers: \t%d\n", n_layers);
    printf("Number of Kernels per Layer:\n");
    

    for (int l = 0; l < n_layers; l++) {
        
        switch (layers[l]->layer_type) {
            case CNN_CONVOLUTION_LAYER: {
                printf("\t Layer %d: %s - %d Kernels\n", l, CNNLayerStrings[layers[l]->layer_type].c_str(), layers[l]->n_kernels);

                for (int k = 0; k < layers[l]->n_kernels; k++) {
                    Kernel* lk = layers[l]->kernels[k];
                    printf("\t\tKernel %d: \n", k);
                    printf("\t\t\t%s(% dx % d) % s Filter\n", lk->getFilterDimensionsString().c_str(), lk->getFilterWidth(), lk->getFilterHeight(), lk->getFilterStyleString().c_str());
                    printf("\t\t\tbias: %lf\n", lk->getBias());
                }

                break;
            }
            case CNN_POOLING_LAYER: {
                printf("Pooling Layer\n");
                break;
            }
            case CNN_FLATTENING_LAYER: {
                printf("Flattening Layer\n");
                break;
            }
            case CNN_NEURAL_NETWORK: {
                printf("Neural Network Layer\n");
                break;
            }
        }
    }
}