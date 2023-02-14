#pragma once
#include <string.h>
#include "CNNLayers.h"

class CNN {
public:
    // TODO: I want to create a CNN based on a 'template' 
    CNN();              
    CNN(ftensor3d, int);            // inputs and n_input_layers
    ~CNN();

    void printCNN();
    int getLayerCount();

    void addNewLayer(CNNLayerType);
    void addNewKernel(FilterStyle);
    void addNewKernel(FixedFilterDimensions);
    void addNewKernel(FixedFilterDimensions, FilterStyle);
    void addNewKernel(DynamicFilterDimensions, int);
    void addNewKernel(DynamicFilterDimensions, int, FilterStyle);

    CNNLayer* getCurrentLayer();
    CNNLayer* getPreviousLayer();


private:
    int n_inputs;           // used to calculate number of samples
    int n_input_layers;     // used to calculate number of layers per samples
    ftensor3d inputs;
    vector<std::string> labels;
    int n_layers;
    vector<CNNLayer*> layers;

    void appendNewLayer(CNNLayer*);
    void appendNewKernel(Kernel*);
    CNNLayer* createNewLayer(CNNLayerType);
    Kernel* createNewKernel();
    Kernel* createNewKernel(FilterStyle);
    Kernel* createNewKernel(FixedFilterDimensions);
    Kernel* createNewKernel(FixedFilterDimensions, FilterStyle);
    Kernel* createNewKernel(DynamicFilterDimensions, int);
    Kernel* createNewKernel(DynamicFilterDimensions, int, FilterStyle);
};