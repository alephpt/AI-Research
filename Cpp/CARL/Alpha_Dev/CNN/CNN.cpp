#include "CNN.h"

CNN::CNN() : n_inputs(0), n_input_layers(0) {}

CNN::~CNN() {
    for (int l = 0; l < n_layers; l++) {
        if (layers[l]->type == CNN_CONVOLUTION_LAYER) {
            getConvolutionalData(layers[l])->kernels.clear();
        }
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

CNNLayer* CNN::getCurrentLayer() { return layers[n_layers - 1]; }
CNNLayer* CNN::getPreviousLayer() { return layers[n_layers - 2]; }
CNNLayerType CNN::getCurrentLayerType() { return layers[n_layers - 1]->type; }
CNNLayerType CNN::getPreviousLayerType() { return layers[n_layers - 2]->type; }
inline ConvolutionLayer* CNN::getConvolutionalData(CNNLayer* layer) { return getConvolutionLayer(layer->data); } 
inline PoolingLayer* CNN::getPoolingData(CNNLayer* layer) { return getPoolingLayer(layer->data); }
Kernel* CNN::createNewKernel() { return new Kernel(); }
Kernel* CNN::createNewKernel(FilterStyle fs) { return new Kernel(fs); }
Kernel* CNN::createNewKernel(FixedFilterDimensions fd) { return new Kernel(fd); }
Kernel* CNN::createNewKernel(FixedFilterDimensions fd, FilterStyle fs) { return new Kernel(fd, fs); }
Kernel* CNN::createNewKernel(DynamicFilterDimensions fd, int n) { return new Kernel(fd, n); }
Kernel* CNN::createNewKernel(DynamicFilterDimensions fd, int n, FilterStyle fs) { return new Kernel(fd, n, fs); }
void CNN::appendNewLayer(CNNLayer* layer) { layers.push_back(layer); n_layers++; }

void CNN::appendNewKernel(Kernel* kernel) { 
    ConvolutionLayer* data = getConvolutionalData(getCurrentLayer());
    data->kernels.push_back(kernel); 
    data->n_kernels++; 
}

CNNLayer* CNN::createNewLayer(CNNLayerType layer_type) {
    CNNLayer* new_layer = new CNNLayer();

    new_layer->type = layer_type;

    return new_layer;
}

void CNN::printCNN() {
    printf("Convolutional Neural Network: \n");
    printf("Number of Inputs: \t\t%d\n", n_inputs);
    printf("Number of Layers per Input: \t%d\n", n_input_layers);
    printf("Number of CNN Layers: \t%d\n", n_layers);
    printf("Number of Kernels per Layer:\n");
    

    for (int l = 0; l < n_layers; l++) {
        
        switch (layers[l]->type) {
            case CNN_CONVOLUTION_LAYER: {
                ConvolutionLayer* cl = getConvolutionLayer(layers[l]->data);
                printf("\t Layer %d: %s - %d Kernels\n", l, CNNLayerStrings[cl->n_kernels].c_str(), cl->n_kernels);

                for (int k = 0; k < cl->n_kernels; k++) {
                    Kernel* lk = cl->kernels[k];
                    printf("\t\tKernel %d: \n", k);
                    printf("\t\t\t%s(% dx % d) %s Filter\n", lk->getFilterDimensionsString().c_str(), lk->getFilterWidth(), lk->getFilterHeight(), lk->getFilterStyleString().c_str());
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
            case CNN_FULLY_CONNECTED: {
                printf("Neural Network Layer\n");
                break;
            }
        }
    }
}