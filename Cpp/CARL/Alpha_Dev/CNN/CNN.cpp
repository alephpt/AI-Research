#include "CNN.h"

CNN::CNN() : n_input_samples(0), n_sample_layers(0), n_layers(0) {}

CNN::~CNN() {
    layers.clear();
    inputs.clear();
    labels.clear();
}

        //////////////////
        // Type Getters //
        //////////////////

CNNLayer* CNN::getCurrentLayer() { return layers[n_layers - 1]; }
CNNLayer* CNN::getPreviousLayer() { return layers[n_layers - 2]; }
CNNLayerType CNN::getCurrentLayerType() { return layers[n_layers - 1]->type; }
CNNLayerType CNN::getPreviousLayerType() { return layers[n_layers - 2]->type; }
Convolution* CNN::getConvolutionalData(CNNLayer* layer) { return std::get<Convolution*>(layer->data); }
Pool* CNN::getPoolingData(CNNLayer* layer) { return std::get<Pool*>(layer->data); }



        /////////////////////////
        // New Layer Functions //
        /////////////////////////

CNNLayer* CNN::createNewLayer(CNNLayerType layer_type) { return new CNNLayer(layer_type); }
void CNN::appendNewLayer(CNNLayer* layer) { layers.push_back(layer); n_layers++; }
void CNN::addNewLayer(CNNLayerType lt) { appendNewLayer(createNewLayer(lt)); }


        //////////////////////////
        // New Kernel Functions //
        //////////////////////////

inline Kernel* CNN::createNewKernel() { return new Kernel(); }
inline Kernel* CNN::createNewKernel(FilterStyle fs) { return new Kernel(fs); }
inline Kernel* CNN::createNewKernel(FixedFilterDimensions fd) { return new Kernel(fd); }
inline Kernel* CNN::createNewKernel(FixedFilterDimensions fd, FilterStyle fs) { return new Kernel(fd, fs); }
inline Kernel* CNN::createNewKernel(DynamicFilterDimensions fd, int n) { return new Kernel(fd, n); }
inline Kernel* CNN::createNewKernel(DynamicFilterDimensions fd, int n, FilterStyle fs) { return new Kernel(fd, n, fs); }

void CNN::appendNewKernel(Kernel* kernel) { 
    Convolution* data = getConvolutionalData(getCurrentLayer());
    data->addNewKernel(kernel); 
}

void CNN::addNewKernel(FilterStyle fs) { appendNewKernel(createNewKernel(fs)); };
void CNN::addNewKernel(FixedFilterDimensions fd) { appendNewKernel(createNewKernel(fd)); };
void CNN::addNewKernel(FixedFilterDimensions fd, FilterStyle fs) { appendNewKernel(createNewKernel(fd, fs)); };
void CNN::addNewKernel(DynamicFilterDimensions fd, int n) { appendNewKernel(createNewKernel(fd, n)); };
void CNN::addNewKernel(DynamicFilterDimensions fd, int n, FilterStyle fs) { appendNewKernel(createNewKernel(fd, n, fs)); };


        ///////////////////////////
        // New Pooling Functions //
        ///////////////////////////

inline Pool* CNN::createNewPool() { return new Pool(); }
inline Pool* CNN::createNewPool(PoolingStyle ps) { return new Pool(ps); }
inline Pool* CNN::createNewPool(FixedFilterDimensions fd) { return new Pool(fd); }
inline Pool* CNN::createNewPool(DynamicFilterDimensions fd, int n) { return new Pool(fd, n); }
inline Pool* CNN::createNewPool(FilterStyle fs) { return new Pool(fs); }
inline Pool* CNN::createNewPool(DynamicFilterDimensions fd, int n, FilterStyle fs) { return new Pool(fd, n, fs); }
inline Pool* CNN::createNewPool(FixedFilterDimensions fd, FilterStyle fs) { return new Pool(fd, fs); }
inline Pool* CNN::createNewPool(PoolingStyle ps, FixedFilterDimensions fd) { return new Pool(ps, fd); }
inline Pool* CNN::createNewPool(PoolingStyle ps, DynamicFilterDimensions fd, int n) { return new Pool(ps, fd, n); }
inline Pool* CNN::createNewPool(PoolingStyle ps, FilterStyle fs) { return new Pool(ps, fs); }
inline Pool* CNN::createNewPool(PoolingStyle ps, FixedFilterDimensions fd, FilterStyle fs) { return new Pool(ps, fd, fs); }
inline Pool* CNN::createNewPool(PoolingStyle ps, DynamicFilterDimensions fd, int n, FilterStyle fs) { return new Pool(ps, fd, n, fs); }

void CNN::setCurrentPoolType(Pool* pool) { getCurrentLayer()->data = pool; }

void CNN::setPoolType() { setCurrentPoolType(createNewPool()); }
void CNN::setPoolType(PoolingStyle ps) { setCurrentPoolType(createNewPool(ps)); }
void CNN::setPoolType(FixedFilterDimensions fd) { setCurrentPoolType(createNewPool(fd)); }
void CNN::setPoolType(DynamicFilterDimensions fd, int n) { setCurrentPoolType(createNewPool(fd, n)); }
void CNN::setPoolType(FilterStyle fs) { setCurrentPoolType(createNewPool(fs)); }
void CNN::setPoolType(DynamicFilterDimensions fd, int n, FilterStyle fs) { setCurrentPoolType(createNewPool(fd, n, fs)); }
void CNN::setPoolType(FixedFilterDimensions fd, FilterStyle fs) { setCurrentPoolType(createNewPool(fd, fs)); }
void CNN::setPoolType(FixedFilterDimensions fd, PoolingStyle ps) { setCurrentPoolType(createNewPool(ps, fd)); }
void CNN::setPoolType(DynamicFilterDimensions fd, int n, PoolingStyle ps) { setCurrentPoolType(createNewPool(ps, fd, n)); }
void CNN::setPoolType(FilterStyle fs, PoolingStyle ps) { setCurrentPoolType(createNewPool(ps, fs)); }
void CNN::setPoolType(FixedFilterDimensions fd, FilterStyle fs, PoolingStyle ps) { setCurrentPoolType(createNewPool(ps, fd, fs)); }
void CNN::setPoolType(DynamicFilterDimensions fd, int n, FilterStyle fs, PoolingStyle ps) { setCurrentPoolType(createNewPool(ps, fd, n, fs)); }


        /////////////////////
        // Layer Functions //
        /////////////////////

ftensor3d CNN::Convolute() { return getConvolutionalData(getCurrentLayer())->convolutionFunction(inputs, n_sample_layers); }
fmatrix CNN::Pooling() { return getPoolingData(getCurrentLayer())->poolingFunction(inputs[0]); }

void CNN::printCNN() {
    printf("Convolutional Neural Network: \n");
    printf("Number of Input Samples: \t%d\n", n_input_samples);
    printf("Number of Layers per Sample: \t%d\n", n_sample_layers);
    printf("Number of CNN Layers: \t\t%d\n", n_layers);
    
    for (int l = 0; l < n_layers; l++) {
        int layer = l + 1;
        switch (layers[l]->type) {
            case CNN_CONVOLUTION_LAYER: {
                Convolution* cl = getConvolutionalData(layers[l]);
                printf("\t Layer %d: %s - %d Kernels\n", layer, CNNLayerStrings[layers[l]->type].c_str(), cl->getKernelCount());

                for (int k = 0; k < cl->getKernelCount(); k++) {
                    Kernel* lk = cl->getKernels()[k];
                    printf("\t\tKernel %d: \n", k);
                    printf("\t\t\t%s(%dx%d) %s Filter\n", lk->getFilterDimensionsString().c_str(), lk->getFilterWidth(), lk->getFilterHeight(), lk->getFilterStyleString().c_str());
                    printf("\t\t\tbias: %lf\n", lk->getBias());
                    printf("\t\t\tstride: %d\n", cl ->getStride());
                }

                break;
            }
            case CNN_POOLING_LAYER: {
                Pool* pl = getPoolingData(layers[l]);
                printf("\t Layer %d: %s\n", layer, CNNLayerStrings[layers[l]->type].c_str());
                printf("\t\t\t%s(%dx%d) %s Filter\n", pl->getFilterDimensionsString().c_str(), pl->getFilterWidth(), pl->getFilterHeight(), pl->getFilterStyleString().c_str());
                printf("\t\t\t%s Format\n", pl->getPoolingStyleString().c_str());
                printf("\t\t\tstride: %d\n", pl->getStride());
                break;
            }
            case CNN_FLATTENING_LAYER: {
                printf("\t Layer %d: %s\n", layer, CNNLayerStrings[layers[l]->type].c_str());
                break;
            }
            case CNN_FULLY_CONNECTED_LAYER: {
                printf("\t Layer %d: %s\n", layer, CNNLayerStrings[layers[l]->type].c_str());
                break;
            }
        }
    }
}

        /////////////////////
        // Input Functions //
        /////////////////////

void CNN::addNewInputSample(ftensor3d input)
{
    if (n_input_samples == 0) {
        n_sample_layers = input.size();
    } else
    if (n_sample_layers != input.size()) {
        printf("[ERROR]: addNewInputSample(ftensor3d) received ismatched sample sizes.\n");
        printf("Hint: Check the number of layers of your input data.");
        return;
    }

    for (int sample = 0; sample < n_sample_layers; sample++) {
        inputs.push_back(input[sample]);
    }

    n_input_samples++;
}
