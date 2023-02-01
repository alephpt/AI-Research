#include "CNN_Tests.h"
#include <vector>
#include <stdio.h>
#include "../Types/General.h"


// Convolution Tests
void testConvolutionInit() {
    int width = 10;
    int height = 17;
    FilterDimensions filter = FIVExFIVE;

    Convolution c = Convolution(height, width, filter);

    printf("test Convolution(%d, %d, %s) initialization\n", height, width, filterString[filter].c_str());
    printf("c.stride: %d\n", c.stride);
    printf("c.input_h: %d\n", c.input_h);
    printf("c.input_w: %d\n", c.input_w);
    c.k->print();
    
}

void testConvolutionFilters() {
    int width = 5;
    int height = 7;
    FilterDimensions filter = ELEVENxELEVEN;

    std::vector<std::vector<float>> input_data = generate2dNoise(height, width);
    Convolution c = Convolution(RELU, height, width, filter);

    c.k->print();

    c.k->setFilterType(NEGATIVE_ASCENDING_FILTER);
    c.k->print();

    c.k->setFilterType(GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterType(INVERSE_GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterType(VERTICAL_GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterType(VERTICAL_INVERSE_GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterType(TOP_LEFT_GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterType(BOTTOM_LEFT_GRADIENT_FILTER);
    c.k->print();

    c.k->setFilterType(GAUSSIAN_FILTER);
    c.k->print();

    c.k->setFilterType(NEGATIVE_GAUSSIAN_FILTER);
    c.k->print();

    c.k->setFilterType(CONICAL_FILTER);
    c.k->print();

    return;
}

void convolve(Convolution* c, FilterStyle filter_type, std::vector<std::vector<float>> input_data, int height, int width) {
    int output_h = 0;
    int output_w = 0;
    
    printf("\n");
    c->k->setFilterType(filter_type);
    c->k->print();

    std::vector<std::vector<float>> output_data = c->convolute(input_data, height, width, &output_h, &output_w);

    printf("\noutput vector - %d x %d\n", output_w, output_h);
    print2DVector(output_data, output_h, output_w);
}

void testConvolutions() {
    int width = 16;
    int height = 20;
    FilterDimensions filter = FIVExFIVE;
    Activation activation_type = TANH;
    FilterStyle filter_type = BOTTOM_LEFT_GRADIENT_FILTER;


    Convolution c = Convolution(activation_type, height, width, filter);
    c.k->setFilterType(filter_type);

    std::vector<std::vector<float>> input_data = generate2dNoise(height, width);
    
    printf("test Convolution(%d, %d, %s)\n", height, width, filterString[filter].c_str());
    printf("c.stride: %d\n", c.stride);
    printf("c.input_h: %d\n", c.input_h);
    printf("c.input_w: %d\n", c.input_w);
    printf("c.k.filter: %s\n", filterStyleString[c.k->getFilterType()].c_str());
    printf("c.k.activation: %s\n", activationString[c.k->getActivationType()].c_str());


    printf("\ninput vector - %d x %d\n", width, height);
    print2DVector(input_data, height, width);

    convolve(&c, BOTTOM_LEFT_GRADIENT_FILTER, input_data, height, width);
    convolve(&c, BALANCED_GAUSSIAN_FILTER, input_data, height, width);
    convolve(&c, CONICAL_FILTER, input_data, height, width);

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

