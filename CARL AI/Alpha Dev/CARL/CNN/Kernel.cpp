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
    populateOffsetFilter, 
    populateAscendingOffsetFilter,
    populateVerticalOffsetFilter,
    populateInverseOffsetFilter,
    populateInverseVerticalOffsetFilter,
    populateTLBRGradientFilter,
    populateBLTRGradientFilter,
    populateGaussianFilter,
    populateNegativeGaussianFilter,
    populateConicalFilter
};


    // Constructor / Destructor
Kernel::Kernel() {
    activation_type = SIGMOID;
    dims = THREExTHREE;
    lookupFilter[dims](&filter.rows, &filter.columns);
    filter.weights = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
    populateFilter(ASCENDING_FILTER);
}

Kernel::Kernel(Activation a) {
    activation_type = a;
    dims = THREExTHREE;
    lookupFilter[dims](&filter.rows, &filter.columns);
    filter.weights = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
    populateFilter(ASCENDING_FILTER);
}

Kernel::Kernel(FilterDimensions f) {
    activation_type = SIGMOID;
    dims = f;
    lookupFilter[dims](&filter.rows, &filter.columns);
    filter.weights = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
    populateFilter(ASCENDING_FILTER);
}

Kernel::Kernel(Activation a, FilterDimensions f) {
    activation_type = a;
    dims = f;
    lookupFilter[dims](&filter.rows, &filter.columns);
    filter.weights = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
    populateFilter(ASCENDING_FILTER);
}

Kernel::Kernel(FilterDimensions f, int n) {
    activation_type = SIGMOID;
    dims = f;
    lookupNFilter[dims](&filter.rows, &filter.columns, n);
    filter.weights = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
    populateFilter(ASCENDING_FILTER);
}

Kernel::Kernel(Activation a, FilterDimensions f, int n) {
    activation_type = a;
    dims = f;
    lookupNFilter[dims](&filter.rows, &filter.columns, n);
    filter.weights = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
    populateFilter(ASCENDING_FILTER);
}


Kernel::~Kernel() { filter.weights.clear(); }


    // public functions
int Kernel::getRows() { return filter.rows; }
int Kernel::getColumns() { return filter.columns; }
Activation Kernel::getActivationType() { return activation_type; }

float Kernel::getProductSum(std::vector<std::vector<float>> input) {
    float sum = 0.0f;

    for (int y = 0; y < filter.rows; y += 1 + stride) {
        for (int x = 0; x < filter.columns; x += 1 + stride) {
            sum += input[y][x] + filter.weights[y][x];
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



void Kernel::populateFilter(FilterStyle s) {
    style = s;
    lookupFilterStyle[style](&filter);
    return;
}

void Kernel::adjustDimensions(FilterDimensions f)
{
    filter.weights.clear();
    dims = f;
    lookupFilter[f](&filter.rows, &filter.columns);
    filter.weights = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
    return;
}

void Kernel::adjustDimensions(FilterDimensions f, int n)
{
    filter.weights.clear();
    dims = f;
    lookupNFilter[f](&filter.rows, &filter.columns, n);
    filter.weights = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
    return;
}

void Kernel::setFilterType(FilterStyle s) {
    style = s;
    populateFilter(style);
    return;
}

void Kernel::setStride(int s) {
    stride = s;
    return;
}

void printColor(float in) {
    if (in < -0.985f) {
        // magenta
        printf("\033[0;35m[%.2lf] \033[0m", in);
    } else
    if (in < -0.7f) {
        // red
        printf("\033[0;31m[%.2lf] \033[0m", in);
    } else
    if (in < -0.35f) {
        // bright red
        printf("\033[0;91m[%.2lf] \033[0m", in);
    } else 
    if (in < -0.11f) {
        // dark yellow
        printf("\033[0;33m[%.2lf] \033[0m", in);
    } else
    if (in < 0.19) {
        // dark green
        printf("\033[0;32m[%.2f] \033[0m", in);
    } else
    if (in < 0.35f) {
        // dark blue
        printf("\033[0;34m[%.2f] \033[0m", in);
    } else
    if (in < 0.7f) {
        // bright blue
        printf("\033[0;94m[%.2f] \033[0m", in);
    } else
    if (in < 0.9f) {
        // bright cyan
        printf("\033[0;96m[%.2f] \033[0m", in);
    } else {
        printf("[%.2f] ", in);
    }
}

void Kernel::print()
{
    printf("%s %s Kernel has %d k->rows and %d k->cols\n", filterString[dims].c_str(), filterStyleString[style].c_str(), filter.rows, filter.columns);
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

