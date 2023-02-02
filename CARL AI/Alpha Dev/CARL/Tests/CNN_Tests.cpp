#include "CNN_Tests.h"
#include <vector>
#include <stdio.h>
#include "../Types/General.h"


// Convolution Tests
void testConvolutionInit() {
    int input_width = 10;
    int input_height = 17;

    CNNLayer c = CNNLayer(input_height, input_width);

    printf("test CNNLayer(%d, %d)\n\n", input_height, input_width);
    printf("\tc.stride: %d\n", c.stride);
    printf("\tc.data->input.height: %d\n", c.data->input.height);
    printf("\tc.data->input.width: %d\n", c.data->input.width);
    printf("\tc.k->getFilterStyle: %s\n", filterStyleString[c.k->getFilterStyle()].c_str());
    printf("\tc.k->getActivationType: %s\n\n", activationString[c.k->getActivationType()].c_str());

    c.k->print();   
}

void testConvolutionFilters() {
    int width = 5;
    int height = 7;
    FilterDimensions filter = ELEVENxELEVEN;

    CNNLayer c = CNNLayer(0, 0, ELEVENxELEVEN);

    c.k->print();

    c.k->setFilterStyle(NEGATIVE_ASCENDING_FILTER);
    c.k->print();

    c.k->setFilterStyle(GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterStyle(INVERSE_GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterStyle(VERTICAL_GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterStyle(VERTICAL_INVERSE_GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterStyle(TOP_LEFT_GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterStyle(BOTTOM_LEFT_GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterStyle(GAUSSIAN_FILTER);
    c.k->print();

    c.k->setFilterStyle(NEGATIVE_GAUSSIAN_FILTER);
    c.k->print();

    c.k->setFilterStyle(BALANCED_GAUSSIAN_FILTER);
    c.k->print();

    c.k->setFilterStyle(CONICAL_FILTER);
    c.k->print();

    return;
}

void convolve(CNNLayer* c, FilterStyle filter_type, ConvolutionType convolution_type) {
    int output_h = 0;
    int output_w = 0;
    
    printf("\n");
    c->k->setFilterStyle(filter_type);
    c->k->print();

    c->convolute(convolution_type);

    CNNSample sample = c->data->layers[c->data->n_layers];
    CNNFeature* feature = sample.features[sample.n_features];

    printf("\noutput vector - %d x %d\n", output_w, output_h);
    print2DVector(feature->values, feature->height, feature->width);
}

void testConvolutions() {
    int input_width = 16;
    int input_height = 20;

    CNNLayer c = CNNLayer(TANH, input_height, input_width, FIVExFIVE);

    c.data->input.values = generate2dNoise(input_height, input_width);
    c.k->setFilterStyle(BOTTOM_LEFT_GRADIENT_FILTER);
    c.setStride(2);

    printf("test CNNLayer(%d, %d)\n\n", input_height, input_width);
    printf("\tc.stride: %d\n", c.stride);
    printf("\tc.data->input.height: %d\n", c.data->input.height);
    printf("\tc.data->input.width: %d\n", c.data->input.width);
    printf("\tc.k->getFilterStyle: %s\n", filterStyleString[c.k->getFilterStyle()].c_str());
    printf("\tc.k->getActivationType: %s\n\n", activationString[c.k->getActivationType()].c_str());

    printf("\ninput vector - %d x %d\n", input_width, input_height);
    print2DVector(c.data->input.values, c.data->input.height, c.data->input.width);

    convolve(&c, INVERSE_GRADIENT_FILTER, STANDARD_CONVOLUTION);
    convolve(&c, VERTICAL_GRADIENT_FILTER, STANDARD_CONVOLUTION);
    convolve(&c, MODIFIED_GAUSSIAN_FILTER, STANDARD_CONVOLUTION);
    convolve(&c, INVERSE_GRADIENT_FILTER, PADDED_CONVOLUTION);
    convolve(&c, VERTICAL_GRADIENT_FILTER, PADDED_CONVOLUTION);
    convolve(&c, MODIFIED_GAUSSIAN_FILTER, PADDED_CONVOLUTION);
    convolve(&c, INVERSE_GRADIENT_FILTER, DILATION_CONVOLUTION);
    convolve(&c, VERTICAL_GRADIENT_FILTER, DILATION_CONVOLUTION);
    convolve(&c, MODIFIED_GAUSSIAN_FILTER, DILATION_CONVOLUTION);

    return;
}

// Pool Tests


// Kernel Tests

static void testKernelFilter(FilterDimensions filter) {
    Kernel* k = new Kernel(filter);
    k->print();
    delete k;
    return;
}

static void testKernelFilter(FilterDimensions filter, int n) {
    Kernel* k = new Kernel(filter, n);
    k->print();
    delete k;
    return;
}

void testKernelInit() {
    testKernelFilter(ELEVENxELEVEN);
    testKernelFilter(SEVENxSEVEN);
    testKernelFilter(FIVExFIVE);
    testKernelFilter(THREExTHREE);
    testKernelFilter(THREExONE);
    testKernelFilter(ONExTHREE);
    testKernelFilter(ONExN, 7);
    testKernelFilter(NxONE, 9);
    testKernelFilter(ONExONE);
    return;
}

void testAdjustKernelDims() {
    Kernel* k = new Kernel(FIVExFIVE);
    k->print();
    k->adjustDimensions(NxONE, 12);
    k->print();
    k->adjustDimensions(ONExN, 7);
    k->print();
    k->adjustDimensions(THREExTHREE);
    k->print();
    delete k;
    return;
}

