#pragma once
#include <string.h>
#include "Kernel.h"
#include "CNNLayers.h"



class CNN {
public:
    // TODO: I want to create a CNN based on a 'template' 
    CNN();              
    ~CNN();

    int getLayerCount();
    void addNewLayer(CNNLayer*);
    CNNLayer* createNewLayer(CNNLayerType);
    CNNLayer* getCurrentLayer();
    CNNLayer* getPreviousLayer();
    void addNewKernel(Kernel*);
    Kernel* createNewKernel();
    Kernel* createNewKernel(FilterStyle);
    Kernel* createNewKernel(FixedFilterDimensions);
    Kernel* createNewKernel(FixedFilterDimensions, FilterStyle);
    Kernel* createNewKernel(DynamicFilterDimensions, int);
    Kernel* createNewKernel(DynamicFilterDimensions, int, FilterStyle);
private:
    int n_inputs;
    int n_input_layers;
    vector<std::string> labels;
    ftensor3d inputs;
    int n_layers;
    vector<CNNLayer*> layers;
};