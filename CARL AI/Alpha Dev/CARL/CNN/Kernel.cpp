#include "Kernel.h"
#include <vector>
#include <stdio.h>

static inline void oneXone(int* r, int* c) { *r = 1; *c = 1; }
static inline void oneXthree(int* r, int* c) { *r = 1; *c = 3; }
static inline void oneXn(int* r, int* c, int n) { *r = 1; *c = n; }
static inline void threeXone(int* r, int* c) { *r = 3; *c = 1; }
static inline void nXone(int* r, int* c, int n) { *r = n; *c = 1; }
static inline void threeXthree(int* r, int* c) { *r = 3; *c = 3; }
static inline void fiveXfive(int* r, int* c) { *r = 5; *c = 5; }
static inline void sevenXseven(int* r, int* c) { *r = 7; *c = 7; }
static inline void elevenXeleven(int* r, int* c) { *r = 11; *c = 11; }

static inline void invalidFilter(int* r, int* c) {
    printf(" [Kernel Filter Error]: Invalid implementation of Kernel Filter for non-N type.\n");
}

static inline void invalidNFilter(int* r, int* c, int n) {
    printf(" [Kernel Filter Error]: Invalid implementation of Kernel Filter for N type.\n");
}


static void (*lookupFilter[])(int* r, int* c) = {
    oneXone, oneXthree, invalidFilter, threeXone, invalidFilter, threeXthree, fiveXfive, sevenXseven, elevenXeleven
};

static void (*lookupNFilter[])(int* r, int* c, int n) = {
    invalidNFilter, invalidNFilter, oneXn, invalidNFilter, nXone, invalidNFilter, invalidNFilter, invalidNFilter, invalidNFilter
};


    // Constructor / Destructor

Kernel::Kernel(FilterDimensions f) {
    dims = f;
    lookupFilter[f](&filter.rows, &filter.columns);
    filter.values = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
}

Kernel::Kernel(FilterDimensions f, int n) {
    dims = f;
    lookupNFilter[f](&filter.rows, &filter.columns, n);
    filter.values = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
}

Kernel::~Kernel() { filter.values.clear(); }


    // public functions

int Kernel::getRows() { return filter.rows; }
int Kernel::getColumns() { return filter.columns; }

float Kernel::getMax(std::vector<std::vector<float>> input)
{
    float max = 0.0f;

    for (int y = 0; y < filter.rows; y++) {
        for (int x = 0; x < filter.columns; x++) {
            float newMax = input[y][x] * filter.values[y][x];
            
            if (max < newMax) { max = newMax; }
        }
    }

    return max;
}

float Kernel::getMaxMean(std::vector<std::vector<float>> input)
{
    float max = 0.0f;

    for (int y = 0; y < filter.rows; y++) {
        for (int x = 0; x < filter.columns; x++) {
            float maxMean = input[y][x] * filter.values[y][x] / 2;
            if (max < maxMean) { max = maxMean; }
        }
    }

    return max;
}

float Kernel::getMeanSum(std::vector<std::vector<float>> input)
{
    float sum = 0.0f;

    for (int y = 0; y < filter.rows; y++) {
        for (int x = 0; x < filter.columns; x++) {
            sum += (input[y][x] + filter.values[y][x]) / 2;
        }
    }

    return sum;
}

float Kernel::getSum(std::vector<std::vector<float>> input)
{
    float sum = 0.0f;

    for (int y = 0; y < filter.rows; y++) {
        for (int x = 0; x < filter.columns; x++) {
            sum += input[y][x] + filter.values[y][x];
        }
    }

    return sum;
}

float Kernel::getMean(std::vector<std::vector<float>> input)
{
    float sum = 0.0f;
    int total = filter.rows + filter.columns;

    for (int y = 0; y < filter.rows; y++) {
        for (int x = 0; x < filter.columns; x++) {
            sum += input[y][x] + filter.values[y][x];
        }
    }

    return sum / total;
}


float Kernel::getSumMean(std::vector<std::vector<float>> input)
{
    float sum = 0.0f;
    int total = filter.rows + filter.columns;

    for (int y = 0; y < filter.rows; y++) {
        for (int x = 0; x < filter.columns; x++) {
            sum += input[y][x] + filter.values[y][x] / 2;
        }
    }

    return sum / total;
}

void Kernel::adjustDimensions(FilterDimensions f)
{
    filter.values.clear();
    dims = f;
    lookupFilter[f](&filter.rows, &filter.columns);
    filter.values = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
}

void Kernel::adjustDimensions(FilterDimensions f, int n)
{
    filter.values.clear();
    dims = f;
    lookupNFilter[f](&filter.rows, &filter.columns, n);
    filter.values = std::vector<std::vector<float>>(filter.rows, std::vector<float>(filter.columns, 0.0f));
}

void Kernel::print()
{
    printf("%s Kernel has %d k->rows and %d k->cols\n", filterString[dims].c_str(), filter.rows, filter.columns);
    for (int y = 0; y < filter.rows; y++) {
        printf("\t\t");
        for (int x = 0; x < filter.columns; x++) {
            printf("[%.2f] ", filter.values[y][x]);
        }
        printf("\n");
    }
    printf("\n");
}

