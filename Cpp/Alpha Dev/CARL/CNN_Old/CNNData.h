
/*
typedef enum {
    POOL_LAYER,
    CONVOLUTION_LAYER,
} LayerType;

typedef struct CNNFeature {
    int index = 0;
    int width = 0;
    int height = 0;
    FilterStyle filter_style = GRADIENT_FILTER;
    fmatrix filter;
    ftensor3d values;
} CNNFeature;

typedef struct CNNSample {
    int n_features = 0;
    LayerType layer;
    FilterDimensions kernel_dimensions;
    Activation activation_type;
    vector<CNNFeature*> features;
} CNNSample;

typedef struct CNNData {
    CNNFeature input;
    CNNFeature output;
    vector<CNNSample*> layers;
    int n_layers = 0;
} CNNData;
*/