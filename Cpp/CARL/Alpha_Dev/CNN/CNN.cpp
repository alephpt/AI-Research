
#include "CNN.h"

CNN::CNN() {
    addNewLayer(createNewLayer(CNN_CONVOLUTION_LAYER));
}

CNN::~CNN() {
    for (int l = 0; l < n_layers; l++) {
        layers[l]->kernels.clear();
    }

    layers.clear();
    inputs.clear();
    labels.clear();
}

int CNN::getLayerCount() { return n_layers; }
void CNN::addNewLayer(CNNLayer* layer) { layers.push_back(layer); n_layers++; }
CNNLayer* CNN::getCurrentLayer() { return layers[n_layers - 1]; }
CNNLayer* CNN::getPreviousLayer() { return layers[n_layers - 2]; }

CNNLayer* CNN::createNewLayer(CNNLayerType layer_type) {
    CNNLayer* new_layer = new CNNLayer;

    new_layer->layer_type = layer_type;

    return new_layer;
}

void CNN::addNewKernel(Kernel* kernel) { getCurrentLayer()->kernels.push_back(kernel); getCurrentLayer()->n_kernels++; }
Kernel* CNN::createNewKernel() { return new Kernel(); }
Kernel* CNN::createNewKernel(FilterStyle fs) { return new Kernel(fs); }
Kernel* CNN::createNewKernel(FixedFilterDimensions fd) { return new Kernel(fd); }
Kernel* CNN::createNewKernel(FixedFilterDimensions fd, FilterStyle fs) { return new Kernel(fd, fs); }
Kernel* CNN::createNewKernel(DynamicFilterDimensions fd, int n) { return new Kernel(fd, n); }
Kernel* CNN::createNewKernel(DynamicFilterDimensions fd, int n, FilterStyle fs) { return new Kernel(fd, n, fs); }
