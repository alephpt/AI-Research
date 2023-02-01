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
    printf("c.padding_x: %d\n", c.padding_x);
    printf("c.padding_y: %d\n", c.padding_y);
    c.k->print();
    
}

void testConvolution() {
    int output_h = 0;
    int output_w = 0;
    int width = 10;
    int height = 10;
    FilterDimensions filter = THREExTHREE;

    std::vector<std::vector<float>> input_data = generate2dNoise(height, width);
    Convolution c = Convolution(height, width, filter);

    printf("test Convolution(%d, %d, %s)\n", height, width, filterString[filter].c_str());
    printf("c.stride: %d\n", c.stride);
    printf("c.input_h: %d\n", c.input_h);
    printf("c.input_w: %d\n", c.input_w);
    printf("c.padding_x: %d\n", c.padding_x);
    printf("c.padding_y: %d\n", c.padding_y);
    c.k->print();

    printf("\ninput vector - %d x %d\n", width, height);
    print2DVector(input_data, height, width);

    std::vector<std::vector<float>> output_data = c.convolute(input_data, height, width, &output_h, &output_w);

    printf("\noutput vector - %d x %d\n", output_w, output_h);
    print2DVector(output_data, output_h, output_w);

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

