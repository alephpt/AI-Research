
/*
typedef enum PoolType {
    MAX_POOLING,
    AVG_POOLING,
    GLOBAL,
    L2_POOLING
} PoolType;

typedef enum ConvolutionType {
    VALID_CONVOLUTION,
    PADDED_CONVOLUTION,
    DILATION_CONVOLUTION,
    PADDED_DILATION_CONVOLUTION
} ConvolutionType;

class CNNLayer {
public:
    CNNLayer(int, int);
    CNNLayer(int, int, int);
    CNNLayer(Activation, int, int);
    CNNLayer(Activation, int, int, int);
    CNNLayer(int, int, FilterDimensions);
    CNNLayer(int, int, int, FilterDimensions);
    CNNLayer(int, int, FilterDimensions, int);
    CNNLayer(int, int, int, FilterDimensions, int);
    CNNLayer(Activation, int, int, FilterDimensions);
    CNNLayer(Activation, int, int, int, FilterDimensions);
    CNNLayer(Activation, int, int, FilterDimensions, int);
    CNNLayer(Activation, int, int, int, FilterDimensions, int);

    void setStride(int);
    void convolute(ConvolutionType, CNNFeature*);
    void printCurrentFeature();
    CNNFeature* createNewFeature();
    CNNFeature* getCurrentFeature();
    CNNSample* getCurrentSample();
    CNNSample* createNewSample();

    int stride;
    Kernel* k;
    CNNData* data;
};
*/