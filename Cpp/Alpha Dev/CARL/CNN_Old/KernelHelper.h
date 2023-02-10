#pragma once

/*
static const fscalar pi = 3.1415926f;


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
    size_t rows = filter->rows;
    size_t cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (fscalar)(x) / ((fscalar)(cols) - 1.0f) * 2.0f - 1.0f;
        }
    }
}

inline void populateInverseGradientFilter(Filter* filter) {
    size_t rows = filter->rows;
    size_t cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = ((fscalar)(x) / ((fscalar)(cols)-1.0f) * 2.0f - 1.0f) * -1.0f;
        }
    }
}

inline void populateVerticalGradientFilter(Filter* filter) {
    size_t rows = filter->rows;
    size_t cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (fscalar)(y) / ((fscalar)(rows) - 1.0f) * 2.0f - 1.0f;
        }
    }
}

inline void populateInverseVerticalGradientFilter(Filter* filter) {
    size_t rows = filter->rows;
    size_t cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = -((fscalar)(y) / ((fscalar)(rows)-1.0f) * 2.0f - 1.0f);
        }
    }
}

inline void populateAscendingFilter(Filter* filter) {
    size_t cols = filter->columns;
    size_t rows = filter->rows;
    size_t units = rows * cols;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (fscalar)(((fscalar)(y * cols + x) / (fscalar)(units)));
        }
    }
}

inline void populateNegativeAscendingFilter(Filter* filter) {
    size_t cols = filter->columns;
    size_t rows = filter->rows;
    size_t units = rows * cols;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (((fscalar)(y * cols + x) / (fscalar)(units)) - 0.49f) * 2.0f;
        }
    }
}


inline void populateTLBRGradientFilter(Filter* filter) {
    size_t rows = filter->rows;
    size_t cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (fscalar)(y) / ((fscalar)(rows) - 1.0f) + (fscalar)(x) / ((fscalar)(cols) - 1.0f) - 1.0f;
        }
    }
}

inline void populateBLTRGradientFilter(Filter* filter) {
    size_t rows = filter->rows;
    size_t cols = filter->columns;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            filter->weights[y][x] = (fscalar)(x) / ((fscalar)(rows) - 1.0f) - (fscalar)(y) / ((fscalar)(cols) - 1.0f);
        }
    }
}

inline void populateGaussianFilter(Filter* filter) {
    size_t rows = filter->rows;
    size_t cols = filter->columns;
    fscalar y_mean = (fscalar)rows / 2.0f;
    fscalar x_mean = (fscalar)cols / 2.0f;
    fscalar y_sigma = (fscalar)rows / 6.0f;
    fscalar x_sigma = (fscalar)cols / 6.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            fscalar dx = (fscalar)x + 0.5f - x_mean;
            fscalar dy = (fscalar)y + 0.5f - y_mean;
            fscalar radial = sqrtf(dx * dx + dy * dy);
            filter->weights[y][x] = (fscalar)expf(-radial * radial / (2 * x_sigma * y_sigma));
        }
    }

    return;
}

inline void populateBalancedGaussianFilter(Filter* filter) {
    size_t rows = filter->rows;
    size_t cols = filter->columns;
    fscalar y_mean = (fscalar)rows / 2.0f;
    fscalar x_mean = (fscalar)cols / 2.0f;
    fscalar y_sigma = (fscalar)rows / 6.0f;
    fscalar x_sigma = (fscalar)cols / 6.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            fscalar dx = (fscalar)x + 0.5f - x_mean;
            fscalar dy = (fscalar)y + 0.5f - y_mean;
            fscalar radial = sqrtf(dx * dx + dy * dy);
            filter->weights[y][x] = ((fscalar)expf(-radial * radial / (2 * x_sigma * y_sigma)) - 0.5f) * 2.0f;
        }
    }

    return;
}

inline void populateNegativeGaussianFilter(Filter* filter) {
    size_t rows = filter->rows;
    size_t cols = filter->columns;
    fscalar y_mean = (fscalar)rows / 2.0f;
    fscalar x_mean = (fscalar)cols / 2.0f;
    fscalar y_sigma = (fscalar)rows / 6.0f;
    fscalar x_sigma = (fscalar)cols / 6.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            fscalar dx = (fscalar)x + 0.5f - x_mean;
            fscalar dy = (fscalar)y + 0.5f - y_mean;
            fscalar radial = sqrtf(dx * dx + dy * dy);
            filter->weights[y][x] = (fscalar)expf(- radial * radial / (2 * x_sigma * y_sigma)) - 1.0f;
        }
    }

    return;
}

inline void populateModifiedGaussianFilter(Filter* filter) {
    size_t rows = filter->rows;
    size_t cols = filter->columns;
    fscalar y_mean = (fscalar)rows / 2.0f;
    fscalar x_mean = (fscalar)cols / 2.0f;
    fscalar y_sigma = (fscalar)rows / 6.0f;
    fscalar x_sigma = (fscalar)cols / 6.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            fscalar dx = (fscalar)x + 0.5f - x_mean;
            fscalar dy = (fscalar)y + 0.5f - y_mean;
            fscalar radial = sqrtf(dx * dx + dy * dy);
            filter->weights[y][x] = ((fscalar)expf(-radial * radial / (2 * x_sigma * y_sigma)) - 0.166f) * -1.55f;
        }
    }

    return;
}

inline void populateConicalFilter(Filter* filter) {
    size_t rows = filter->rows;
    size_t cols = filter->columns;
    fscalar y_mean = (fscalar)rows / 2.0f;
    fscalar x_mean = (fscalar)cols / 2.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            fscalar dx = (fscalar)(x);
            fscalar dy = (fscalar)(y) - y_mean;
            fscalar theta = atan2f(dy, dx);
            filter->weights[y][x] = ((theta + pi) / pi - 1.0f) * 1.9f;
        }
    }

    return;
}
*/