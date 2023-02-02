#include "Kernel.h"
#include "KernelHelper.h"
#include <vector>
#include <stdio.h>
#include <math.h>


static void (*lookupFilter[])(int* r, int* c) = {
    oneXone, oneXthree, invalidFilter, threeXone, invalidFilter, threeXthree, fiveXfive, sevenXseven, elevenXeleven
};

static void (*lookupNFilter[])(int* r, int* c, int n) = {
    invalidNFilter, invalidNFilter, oneXn, invalidNFilter, nXone, invalidNFilter, invalidNFilter, invalidNFilter, invalidNFilter
};

static void (*lookupFilterStyle[])(Filter* filter) = {
    populateAscendingFilter,
    populateNegativeAscendingFilter,
    populateGradientFilter,
    populateVerticalGradientFilter,
    populateInverseGradientFilter,
    populateInverseVerticalGradientFilter,
    populateTLBRGradientFilter,
    populateBLTRGradientFilter,
    populateGaussianFilter,
    populateBalancedGaussianFilter,
    populateNegativeGaussianFilter,
    populateModifiedGaussianFilter,
    populateConicalFilter
};


    // Constructor / Destructor
Kernel::Kernel() {
    activation_type = RELU;
    style = GRADIENT_FILTER;
    setFilterDimensions(THREExTHREE);
}

Kernel::Kernel(Activation a) {
    activation_type = a;
    style = GRADIENT_FILTER;
    setFilterDimensions(THREExTHREE);
}

Kernel::Kernel(FilterStyle fs) {
    activation_type = RELU;
    style = fs;
    setFilterDimensions(THREExTHREE);
}

Kernel::Kernel(FilterDimensions fd) {
    activation_type = RELU;
    style = GRADIENT_FILTER;
    setFilterDimensions(fd);
}

Kernel::Kernel(Activation a, FilterStyle fs) {
    activation_type = a;
    style = fs;
    setFilterDimensions(THREExTHREE);
}

Kernel::Kernel(FilterDimensions fd, FilterStyle fs) {
    activation_type = RELU;
    style = fs;
    setFilterDimensions(fd);
}

Kernel::Kernel(Activation a, FilterDimensions fd) {
    activation_type = a;
    style = GRADIENT_FILTER;
    setFilterDimensions(fd);
}

Kernel::Kernel(Activation a, FilterDimensions fd, FilterStyle fs) {
    activation_type = a;
    style = fs;
    setFilterDimensions(fd);
}

Kernel::Kernel(FilterDimensions fd, int n) {
    activation_type = RELU;
    style = GRADIENT_FILTER;
    setFilterDimensions(fd, n);
}

Kernel::Kernel(FilterDimensions fd, int n, FilterStyle fs) {
    activation_type = RELU;
    style = fs;
    setFilterDimensions(fd, n);
}

Kernel::Kernel(Activation a, FilterDimensions fd, int n) {
    activation_type = a;
    style = GRADIENT_FILTER;
    setFilterDimensions(fd, n);
}

Kernel::Kernel(Activation a, FilterDimensions fd, int n, FilterStyle fs) {
    activation_type = a;
    style = fs;
    setFilterDimensions(fd, n);
}

Kernel::~Kernel() { filter.weights.clear(); }

    // public functions
void Kernel::setFilterDimensions(FilterDimensions f) { 
    dimensions = f; 
    lookupFilter[dimensions](&filter.rows, &filter.columns); 
    filter.weights = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
    populateFilter(style); 
}

void Kernel::setFilterDimensions(FilterDimensions f, int n) { 
    dimensions = f; 
    lookupNFilter[dimensions](&filter.rows, &filter.columns, n); 
    filter.weights = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
    populateFilter(style); 
}

void Kernel::populateFilter(FilterStyle s) {
    style = s;
    lookupFilterStyle[style](&filter);
    return;
}

void Kernel::adjustDimensions(FilterDimensions f)
{
    filter.weights.clear();
    setFilterDimensions(f);
}

void Kernel::adjustDimensions(FilterDimensions f, int n)
{
    filter.weights.clear();
    setFilterDimensions(f, n);
}

void Kernel::setStride(int s) { stride = s; }
void Kernel::setActivationType(Activation a) { activation_type = a; }
void Kernel::setFilterStyle(FilterStyle s) { style = s; populateFilter(style); }
std::vector<std::vector<float>> Kernel::getWeights() { return filter.weights; }
FilterDimensions Kernel::getDimensions() { return dimensions; }
Activation Kernel::getActivationType() { return activation_type; }
FilterStyle Kernel::getFilterStyle() { return style; }
int Kernel::getColumns() { return filter.columns; }
int Kernel::getRows() { return filter.rows; }


float Kernel::getProductSum(std::vector<std::vector<float>> input) {
    float sum = 0.0f;

    for (int y = 0; y < filter.rows; y += 1 + stride) {
        for (int x = 0; x < filter.columns; x += 1 + stride) {
            sum += input[y][x] * filter.weights[y][x];
        }
    }

    return sum;
}

float Kernel::getMax(std::vector<std::vector<float>> input)
{
    float max = 0.0f;

    for (int y = 0; y < filter.rows; y += 1 + stride) {
        for (int x = 0; x < filter.columns; x += 1 + stride) {
            float newMax = input[y][x];
            
            if (newMax > max) { max = newMax; }
        }
    }

    return max;
}



float Kernel::getMaxMean(std::vector<std::vector<float>> input)
{
    float max = 0.0f;

    for (int y = 0; y < filter.rows; y += 1 + stride) {
        for (int x = 0; x < filter.columns; x += 1 + stride) {
            float maxMean = input[y][x] + filter.weights[y][x] / 2;
            if (max < maxMean) { max = maxMean; }
        }
    }

    return max;
}



float Kernel::getSum(std::vector<std::vector<float>> input)
{
    float sum = 0.0f;

    for (int y = 0; y < filter.rows; y += 1 + stride) {
        for (int x = 0; x < filter.columns; x += 1 + stride) {
            sum += input[y][x] + filter.weights[y][x];
        }
    }

    return sum;
}

float Kernel::getSumMean(std::vector<std::vector<float>> input)
{
    float sum = 0.0f;
    int total = filter.rows + filter.columns;

    for (int y = 0; y < filter.rows; y += 1 + stride) {
        for (int x = 0; x < filter.columns; x += 1 + stride) {
            sum += input[y][x] + filter.weights[y][x];
        }
    }

    return sum / total;
}

float Kernel::getMeanSum(std::vector<std::vector<float>> input)
{
    float sum = 0.0f;

    for (int y = 0; y < filter.rows; y += 1 + stride) {
        for (int x = 0; x < filter.columns; x += 1 + stride) {
            sum += (input[y][x] + filter.weights[y][x]) / 2;
        }
    }

    return sum;
}

float Kernel::getMean(std::vector<std::vector<float>> input)
{
    float sum = 0.0f;
    int total = filter.rows + filter.columns;

    for (int y = 0; y < filter.rows; y += 1 + stride) {
        for (int x = 0; x < filter.columns; x += 1 + stride) {
            sum += (input[y][x] + filter.weights[y][x]) / 2;
        }
    }

    return sum / total;
}

void Kernel::print()
{
    printf("%s %s Kernel\n", filterString[dimensions].c_str(), filterStyleString[style].c_str());
    for (int y = 0; y < filter.rows; y++) {
        printf("\t\t");
        for (int x = 0; x < filter.columns; x++) {
            printColor(filter.weights[y][x]);
        }
        printf("\n");
    }
    printf("\n");
    return;
}

