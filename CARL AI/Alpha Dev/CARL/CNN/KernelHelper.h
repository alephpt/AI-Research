#pragma once
#include <vector>
#include "Kernel.h"
#include <math.h>

static const float pi = 3.1415926;


    // FILTER DIMENSIONS //
inline void oneXone(int* r, int* c) { *r = 1; *c = 1; }
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
inline void populateOffsetFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;
    float y_mean = (float)rows / 2.0f;
    float x_mean = (float)cols / 2.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (float)(x - x_mean + 0.50f) * 
                                    ((expf(-0.5 * ((float)x + 0.5f - y_mean) * ((float)x + 0.5f - y_mean) + 
                                                  ((float)y + 0.5f - x_mean) * ((float)y + 0.5f - x_mean)) / 
                                                  (2 * pi * x_mean * y_mean)) / 3.0f);
        }
    }
}

inline void populateInverseOffsetFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;
    float y_median = (float)rows / 2.0f;
    float x_median = (float)cols / 2.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = ((float)(x - x_median + 0.5f) * -1.0f);
        }
    }
}

inline void populateVerticalOffsetFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;
    float y_mean = (float)rows / 2.0f;
    float x_mean = (float)cols / 2.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (float)(y - y_mean + 0.50f) * 
                                    ((expf(-0.5 * ((float)x + 0.5f - x_mean) * ((float)x + 0.5f - x_mean) + 
                                                  ((float)y + 0.5f - y_mean) * ((float)y + 0.5f - y_mean)) / 
                                                  (2 * pi * x_mean * y_mean)) / 3.0f);
        }
    }
}

inline void populateInverseVerticalOffsetFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;
    float y_mean = (float)rows / 2.0f;
    float x_mean = (float)cols / 2.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (float)(y - y_mean + 0.50f) * 
                                    ((expf(-0.5 * ((float)x + 0.5f - x_mean) * ((float)x + 0.5f - x_mean) + 
                                                  ((float)y + 0.5f - y_mean) * ((float)y + 0.5f - y_mean)) / 
                                                  (2 * pi * x_mean * y_mean)) / 3.0f) * -1.0f;
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

inline void populateAscendingOffsetFilter(Filter* filter) {
    int cols = filter->columns;
    int rows = filter->rows;
    int units = rows * cols;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (float)(((float)(y * cols + x) / (float)(units))) - 0.48f;
        }
    }
}


inline void populateGradientFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;
    float y_median = (float)rows / 2.0f;
    float x_median = (float)cols / 2.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = 0.1f;
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

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float dx = (float)j + 0.5f - x_mean;
            float dy = (float)i + 0.5f - y_mean;
            float radial = sqrtf(dx * dx + dy * dy);
            filter->weights[i][j] = 2.0f * expf(- radial * radial / (2 * x_sigma * y_sigma)) - 0.5f;
        }
    }

    return;
}

inline void populateAngledGradientFilter(Filter* filter) {
    int rows = filter->rows;
    int cols = filter->columns;
    float scale = 0.0f;
    float y_mean = (float)rows / 2.0f;
    float x_mean = (float)cols / 2.0f;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i + j == 0) { scale = ((expf(-0.5 * ((float)j + 0.5f - x_mean) * ((float)i + 0.5f - x_mean) + 
                                                  ((float)i + 0.5f - y_mean) * ((float)j + 0.5f - y_mean)))); }
            filter->weights[i][j] = ((expf(-0.5 * ((float)j + 0.5f - x_mean) * ((float)i + 0.5f - x_mean) + 
                                                  ((float)i + 0.5f - y_mean) * ((float)j + 0.5f - y_mean))) / scale ); 
        }
    }

    return;
}