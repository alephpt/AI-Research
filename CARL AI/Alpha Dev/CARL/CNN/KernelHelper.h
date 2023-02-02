#pragma once
#include <vector>
#include "Kernel.h"
#include "../Types/General.h"
#include <math.h>

static const float pi = 3.1415926f;


    // FILTER DIMENSIONS //
inline void oneXone(int* r, int* c) { *r = 1; *c = 1; }
inline void twoXtwo(int* r, int* c) { *r = 2; *c = 2; }
inline void oneXthree(int* r, int* c) { *r = 1; *c = 3; }
inline void oneXn(int* r, int* c, int n) { *r = 1; *c = n; }
inline void threeXone(int* r, int* c) { *r = 3; *c = 1; }
inline void nXone(int* r, int* c, int n) { *r = n; *c = 1; }
inline void threeXthree(int* r, int* c) { *r = 3; *c = 3; }
inline void fiveXfive(int* r, int* c) { *r = 5; *c = 5; }
inline void sevenXseven(int* r, int* c) { *r = 7; *c = 7; }
inline void elevenXeleven(int* r, int* c) { *r = 11; *c = 11; }

inline void invalidFilter(int* r, int* c) {
    printf(" [Kernel Filter Error]: Invalid implementation of Kernel Filter for non-N type.\n");
}

inline void invalidNFilter(int* r, int* c, int n) {
    printf(" [Kernel Filter Error]: Invalid implementation of Kernel Filter for N type.\n");
}


    // FILTER PATTERNS
inline void populateGradientFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (float)(x) / ((float)(cols) - 1.0f) * 2.0f - 1.0f;
        }
    }
}

inline void populateInverseGradientFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = ((float)(x) / ((float)(cols)-1.0f) * 2.0f - 1.0f) * -1.0f;
        }
    }
}

inline void populateVerticalGradientFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (float)(y) / ((float)(rows) - 1.0f) * 2.0f - 1.0f;
        }
    }
}

inline void populateInverseVerticalGradientFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = -((float)(y) / ((float)(rows)-1.0f) * 2.0f - 1.0f);
        }
    }
}

inline void populateAscendingFilter(Filter* filter) {
    int cols = filter->columns;
    int rows = filter->rows;
    int units = rows * cols;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (float)(((float)(y * cols + x) / (float)(units)));
        }
    }
}

inline void populateNegativeAscendingFilter(Filter* filter) {
    int cols = filter->columns;
    int rows = filter->rows;
    int units = rows * cols;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (((float)(y * cols + x) / (float)(units)) - 0.49f) * 2.0f;
        }
    }
}


inline void populateTLBRGradientFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (float)(y) / ((float)(rows) - 1.0f) + (float)(x) / ((float)(cols) - 1.0f) - 1.0f;
        }
    }
}

inline void populateBLTRGradientFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (float)(x) / ((float)(rows) - 1.0f) - (float)(y) / ((float)(cols) - 1.0f);
        }
    }
}

inline void populateGaussianFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;
    float y_mean = (float)rows / 2.0f;
    float x_mean = (float)cols / 2.0f;
    float y_sigma = (float)rows / 6.0f;
    float x_sigma = (float)cols / 6.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            float dx = (float)x + 0.5f - x_mean;
            float dy = (float)y + 0.5f - y_mean;
            float radial = sqrtf(dx * dx + dy * dy);
            filter->weights[y][x] = (float)expf(-radial * radial / (2 * x_sigma * y_sigma));
        }
    }

    return;
}

inline void populateBalancedGaussianFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;
    float y_mean = (float)rows / 2.0f;
    float x_mean = (float)cols / 2.0f;
    float y_sigma = (float)rows / 6.0f;
    float x_sigma = (float)cols / 6.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            float dx = (float)x + 0.5f - x_mean;
            float dy = (float)y + 0.5f - y_mean;
            float radial = sqrtf(dx * dx + dy * dy);
            filter->weights[y][x] = ((float)expf(-radial * radial / (2 * x_sigma * y_sigma)) - 0.5f) * 2.0f;
        }
    }

    return;
}

inline void populateNegativeGaussianFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;
    float y_mean = (float)rows / 2.0f;
    float x_mean = (float)cols / 2.0f;
    float y_sigma = (float)rows / 6.0f;
    float x_sigma = (float)cols / 6.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            float dx = (float)x + 0.5f - x_mean;
            float dy = (float)y + 0.5f - y_mean;
            float radial = sqrtf(dx * dx + dy * dy);
            filter->weights[y][x] = (float)expf(- radial * radial / (2 * x_sigma * y_sigma)) - 1.0f;
        }
    }

    return;
}

inline void populateModifiedGaussianFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;
    float y_mean = (float)rows / 2.0f;
    float x_mean = (float)cols / 2.0f;
    float y_sigma = (float)rows / 6.0f;
    float x_sigma = (float)cols / 6.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            float dx = (float)x + 0.5f - x_mean;
            float dy = (float)y + 0.5f - y_mean;
            float radial = sqrtf(dx * dx + dy * dy);
            filter->weights[y][x] = ((float)expf(-radial * radial / (2 * x_sigma * y_sigma)) - 0.166f) * -1.55f;
        }
    }

    return;
}

inline void populateConicalFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;
    float y_mean = (float)rows / 2.0f;
    float x_mean = (float)cols / 2.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            float dx = (float)(x);
            float dy = (float)(y) - y_mean;
            float theta = atan2f(dy, dx);
            filter->weights[y][x] = ((theta + pi) / pi - 1.0f) * 1.9f;
        }
    }

    return;
}