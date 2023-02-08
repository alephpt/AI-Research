#include "CNN_Tests.h"
#include <vector>
#include <stdio.h>
#include "../Tests/TestData.h"
#include "../Data/Load_Data.h"

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

    c.k->printFilter();   
}

void testConvolutionInput() {
    int input_width = (int)smiles[0].size();
    int input_height = (int)smiles.size();

    CNNLayer c = CNNLayer(input_height, input_width);
    
    c.data->input.values = ftensor3d(4, smiles);

    printf("test CNNLayer(%d, %d)\n\n", input_height, input_width);
    printf("\tc.stride: %d\n", c.stride);
    printf("\tc.data->input.height: %d\n", c.data->input.height);
    printf("\tc.data->input.width: %d\n", c.data->input.width);
    printf("\tc.k->getFilterStyle: %s\n", filterStyleString[c.k->getFilterStyle()].c_str());
    printf("\tc.k->getActivationType: %s\n\n", activationString[c.k->getActivationType()].c_str());

    printf("\ninput vector - %d x %d\n", input_width, input_height);
    printFMatrix(c.data->input.values);

    c.k->printFilter();
}


void testConvolutionFilters() {
    int width = 5;
    int height = 7;
    FilterDimensions filter = ELEVENxELEVEN;

    CNNLayer c = CNNLayer(0, 0, ELEVENxELEVEN);

    c.k->printFilter();

    c.k->setFilterStyle(NEGATIVE_ASCENDING_FILTER);
    c.k->printFilter();

    c.k->setFilterStyle(GRADIENT_FILTER);
    c.k->printFilter();

    c.k->setFilterStyle(INVERSE_GRADIENT_FILTER);
    c.k->printFilter();

    c.k->setFilterStyle(VERTICAL_GRADIENT_FILTER);
    c.k->printFilter();

    c.k->setFilterStyle(VERTICAL_INVERSE_GRADIENT_FILTER);
    c.k->printFilter();

    c.k->setFilterStyle(TOP_LEFT_GRADIENT_FILTER);
    c.k->printFilter();

    c.k->setFilterStyle(BOTTOM_LEFT_GRADIENT_FILTER);
    c.k->printFilter();

    c.k->setFilterStyle(GAUSSIAN_FILTER);
    c.k->printFilter();

    c.k->setFilterStyle(NEGATIVE_GAUSSIAN_FILTER);
    c.k->printFilter();

    c.k->setFilterStyle(BALANCED_GAUSSIAN_FILTER);
    c.k->printFilter();

    c.k->setFilterStyle(CONICAL_FILTER);
    c.k->printFilter();

    return;
}

void convolve(CNNLayer* c, FilterStyle filter_type, ConvolutionType convolution_type) {
    printf("\n");

    c->k->setFilterStyle(filter_type);
    c->k->printFilter();

    c->convolute(convolution_type, &c->data->input);
    c->convolute(convolution_type, c->getCurrentFeature());
    c->convolute(convolution_type, c->getCurrentFeature());
    c->convolute(convolution_type, c->getCurrentFeature());
    c->convolute(convolution_type, c->getCurrentFeature());
    c->convolute(convolution_type, c->getCurrentFeature());
    c->printCurrentFeature();
}

void testConvolutions() {
    Image img = Image("Data/Image/Samples/sample_1.jpeg");
    int input_width = img.getWidth();
    int input_height = img.getHeight();
    ftensor3d img_data = img.getTensor();

    CNNLayer c = CNNLayer(RELU, input_height, input_width, THREExTHREE);

    c.data->input.values = img_data[4];
    
    printf("test CNNLayer(%d, %d)\n\n", input_height, input_width);
    printf("\tc.stride: %d\n", c.stride);
    printf("\tc.data->input.height: %d\n", c.data->input.height);
    printf("\tc.data->input.width: %d\n", c.data->input.width);
    printf("\tc.k->getFilterStyle: %s\n", filterStyleString[c.k->getFilterStyle()].c_str());
    printf("\tc.k->getActivationType: %s\n\n", activationString[c.k->getActivationType()].c_str());

//    printf("\ninput vector - %d x %d\n", input_width, input_height);
//    printFMatrix(c.data->input.values);

    convolve(&c, GRADIENT_FILTER, DILATION_CONVOLUTION);
    // convolve(&c, VERTICAL_GRADIENT_FILTER, VALID_CONVOLUTION);
    // convolve(&c, MODIFIED_GAUSSIAN_FILTER, VALID_CONVOLUTION);
    // convolve(&c, INVERSE_GRADIENT_FILTER, PADDED_CONVOLUTION);
    // convolve(&c, VERTICAL_GRADIENT_FILTER, PADDED_CONVOLUTION);
    // convolve(&c, MODIFIED_GAUSSIAN_FILTER, PADDED_CONVOLUTION);

    // convolve(&c, INVERSE_GRADIENT_FILTER, DILATION_CONVOLUTION);
    // convolve(&c, VERTICAL_GRADIENT_FILTER, DILATION_CONVOLUTION);
    // convolve(&c, MODIFIED_GAUSSIAN_FILTER, DILATION_CONVOLUTION);
    // convolve(&c, INVERSE_GRADIENT_FILTER, PADDED_DILATION_CONVOLUTION);
    // convolve(&c, VERTICAL_GRADIENT_FILTER, PADDED_DILATION_CONVOLUTION);
    // convolve(&c, MODIFIED_GAUSSIAN_FILTER, PADDED_DILATION_CONVOLUTION);

    return;
}

// Pool Tests


// Kernel Tests

static void testKernelFilter(FilterDimensions filter) {
    Kernel* k = new Kernel(filter);
    k->printFilter();
    delete k;
    return;
}

static void testKernelFilter(FilterDimensions filter, int n) {
    Kernel* k = new Kernel(filter, n);
    k->printFilter();
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
    k->printFilter();
    k->adjustDimensions(NxONE, 12);
    k->printFilter();
    k->adjustDimensions(ONExN, 7);
    k->printFilter();
    k->adjustDimensions(THREExTHREE);
    k->printFilter();
    delete k;
    return;
}

