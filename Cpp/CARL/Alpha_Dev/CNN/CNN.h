#pragma once
#include <string.h>
#include "CNNLayers.h"

class CNN {
public:
    // TODO: I want to create a CNN based on a 'template' 
    CNN();              
    ~CNN();

    void printCNN();

    void addNewInputSample(ftensor3d);

    void addNewLayer(CNNLayerType);

    void addNewKernel(FilterStyle);
    void addNewKernel(FixedFilterDimensions);
    void addNewKernel(FixedFilterDimensions, FilterStyle);
    void addNewKernel(DynamicFilterDimensions, int);
    void addNewKernel(DynamicFilterDimensions, int, FilterStyle);

    void setPoolType();
    void setPoolType(PoolingStyle);
    void setPoolType(FilterStyle);
    void setPoolType(FixedFilterDimensions);
    void setPoolType(FixedFilterDimensions, FilterStyle);
    void setPoolType(DynamicFilterDimensions, int);
    void setPoolType(DynamicFilterDimensions, int, FilterStyle);
    void setPoolType(FilterStyle, PoolingStyle);
    void setPoolType(FixedFilterDimensions, PoolingStyle);
    void setPoolType(FixedFilterDimensions, FilterStyle, PoolingStyle);
    void setPoolType(DynamicFilterDimensions, int, PoolingStyle);
    void setPoolType(DynamicFilterDimensions, int, FilterStyle, PoolingStyle);

    ftensor3d Convolute();
    fmatrix Pooling();
private:
    int n_input_samples;        // defined by inputs / n_sample_layers
    int n_sample_layers;        // used to calculate number of layers per samples  - CMYK
    ftensor3d inputs = ftensor3d(0, fmatrix(0, vector<fscalar>(0)));
    vector<std::string> labels;
    int n_layers;
    vector<CNNLayer*> layers;


    CNNLayer* getCurrentLayer();
    CNNLayer* getPreviousLayer();
    CNNLayerType getCurrentLayerType();
    CNNLayerType getPreviousLayerType();
    Convolution* getConvolutionalData(CNNLayer* layer);
    Pool* getPoolingData(CNNLayer* layer);

    void appendNewLayer(CNNLayer*);
    CNNLayer* createNewLayer(CNNLayerType);

    void appendNewKernel(Kernel*);
    inline Kernel* createNewKernel();
    inline Kernel* createNewKernel(FilterStyle);
    inline Kernel* createNewKernel(FixedFilterDimensions);
    inline Kernel* createNewKernel(FixedFilterDimensions, FilterStyle);
    inline Kernel* createNewKernel(DynamicFilterDimensions, int);
    inline Kernel* createNewKernel(DynamicFilterDimensions, int, FilterStyle);

    void setCurrentPoolType(Pool*);
    inline Pool* createNewPool();
    inline Pool* createNewPool(PoolingStyle);
    inline Pool* createNewPool(FilterStyle);
    inline Pool* createNewPool(FixedFilterDimensions);
    inline Pool* createNewPool(FixedFilterDimensions, FilterStyle);
    inline Pool* createNewPool(DynamicFilterDimensions, int);
    inline Pool* createNewPool(DynamicFilterDimensions, int, FilterStyle);
    inline Pool* createNewPool(PoolingStyle, FilterStyle);
    inline Pool* createNewPool(PoolingStyle, FixedFilterDimensions);
    inline Pool* createNewPool(PoolingStyle, FixedFilterDimensions, FilterStyle);
    inline Pool* createNewPool(PoolingStyle, DynamicFilterDimensions, int);
    inline Pool* createNewPool(PoolingStyle, DynamicFilterDimensions, int, FilterStyle);
};