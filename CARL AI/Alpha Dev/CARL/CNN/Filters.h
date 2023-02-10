#pragma once
#include "../Types/Types.h"

typedef struct Filter {
    fmatrix weights;
    int width = 0;
    int height = 0;
} Filter;

inline void oneXone(int* r, int* c) { *r = 1; *c = 1; }
inline void twoXtwo(int* r, int* c) { *r = 2; *c = 2; }
inline void threeXthree(int* r, int* c) { *r = 3; *c = 3; }
inline void fiveXfive(int* r, int* c) { *r = 5; *c = 5; }
inline void sevenXseven(int* r, int* c) { *r = 7; *c = 7; }
inline void elevenXeleven(int* r, int* c) { *r = 11; *c = 11; }
inline void oneXn(int* r, int* c, int n) { *r = 1; *c = n; }
inline void twoXn(int* r, int* c, int n) { *r = 2; *c = n; }
inline void threeXn(int* r, int* c, int n) { *r = 3; *c = n; }
inline void nXone(int* r, int* c, int n) { *r = n; *c = 1; }
inline void nXtwo(int* r, int* c, int n) { *r = n; *c = 2; }
inline void nXthree(int* r, int* c, int n) { *r = n; *c = 3; }

inline void createRightEdgeFilter(Filter* f) {
    if (f->width < 2) {
        for (int y = 0; y < f->height; y++) {
            f->weights[y][0] = 1.0f;
        }
        return;
    }

    for (int y = 0; y < f->height; y++) {
        f->weights[y][f->width - 2] = -1.0f;
        f->weights[y][f->width - 1] = 1.0f;
    }
}

inline void createLeftEdgeFilter(Filter* f) {
    if (f->width < 2) {
        for (int y = 0; y < f->height; y++) {
            f->weights[y][0] = 1.0f;
        }
        return;
    }

    for (int y = 0; y < f->height; y++) {
        f->weights[y][0] = 1.0f;
        f->weights[y][1] = -1.0f;
    }
}

inline void createTopEdgeFilter(Filter* f) {
    if (f->height < 2) {
        for (int x = 0; x < f->width; x++) {
            f->weights[0][x] = 1.0f;
        }
        return;
    }

    for (int x = 0; x < f->width; x++) {
        f->weights[0][x] = 1.0f;
        f->weights[1][x] = -1.0f;
    }
} 

inline void createBottomEdgeFilter(Filter* f) {
    if (f->height < 2) {
        for (int x = 0; x < f->width; x++) {
            f->weights[0][x] = 1.0f;
        }
        return;
    }

    for (int x = 0; x < f->width; x++) {
        f->weights[f->height - 1][x] = 1.0f;
        f->weights[f->height - 2][x] = -1.0f;
    }
}

inline void createTopRightCornerFilter(Filter* f) {
    if (f->width < 2) {
        f->weights[0][0] = 1.0f;
        f->weights[1][0] = -1.0f;
        return;
    }
    
    if (f->height < 2) {
        f->weights[0][f->width - 1] = 1.0f;
        f->weights[0][f->width - 2] = -1.0f;
        return;
    }

    for (int x = 0; x < f->width; x++) {
        f->weights[0][x] = 1.0f;
        f->weights[1][x] = -1.0f;
    }

    for (int y = 1; y < f->height; y++) {
        f->weights[y][f->width - 1] = 1.0f;
        f->weights[y][f->width - 2] = -1.0f;
    }
}

inline void createBottomRightCornerFilter(Filter* f) {
    if (f->width < 2) {
        f->weights[f->height - 1][0] = 1.0f;
        f->weights[f->height - 2][0] = -1.0f;
        return;
    }

    if (f->height < 2) {
        f->weights[0][f->width - 1] = 1.0f;
        f->weights[0][f->width - 2] = -1.0f;
        return;
    }

    for (int x = 0; x < f->width; x++) {
        f->weights[f->height - 1][x] = 1.0f;
        f->weights[f->height - 2][x] = -1.0f;
    }

    for (int y = 0; y < f->height - 1; y++) {
        f->weights[y][f->width - 1] = 1.0f;
        f->weights[y][f->width - 2] = -1.0f;
    }
}

inline void createBottomLeftCornerFilter(Filter* f) {
    if (f->width < 2) {
        f->weights[f->height - 1][0] = 1.0f;
        f->weights[f->height - 2][0] = -1.0f;
        return;
    }

    if (f->height < 2) {
        f->weights[0][0] = 1.0f;
        f->weights[0][1] = -1.0f;
        return;
    }

    for (int x = 0; x < f->width; x++) {
        f->weights[f->height - 1][x] = 1.0f;
        f->weights[f->height - 2][x] = -1.0f;
    }

    for (int y = 0; y < f->height - 1; y++) {
        f->weights[y][0] = 1.0f;
        f->weights[y][1] = -1.0f;
    }
}

inline void createTopLeftCornerFilter(Filter* f) {
    if (f->width < 2) {
        f->weights[0][0] = 1.0f;
        f->weights[1][0] = -1.0f;
        return;
    }

    if (f->height < 2) {
        f->weights[0][0] = 1.0f;
        f->weights[0][1] = -1.0f;
        return;
    }

    for (int x = 0; x < f->width; x++) {
        f->weights[0][x] = 1.0f;
        f->weights[1][x] = -1.0f;
    }

    for (int y = 1; y < f->height; y++) {
        f->weights[y][0] = 1.0f;
        f->weights[y][1] = -1.0f;
    }
}

inline void createAscendingFilter(Filter* f) {
    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            f->weights[y][x] =  2.0f * (float)(y * f->width + x) / (float)(f->width * f->height - 1.0f) - 1.0f;
        }
    }
}

inline void createDescendingFilter(Filter* f) {
    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            f->weights[y][x] = -(2.0f * (float)(y * f->width + x) / (float)(f->width * f->height - 1.0f)) + 1.0f;
        }
    }
}

inline void createVerticalAscendingFilter(Filter* f) {
    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            f->weights[y][x] = -1.0f + 2.0f * ((float)(y + x * f->width) / (float)(f->height * f->width - 1));
        }
    }
}

inline void createVerticalDescendingFilter(Filter* f) {
    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            f->weights[y][x] = -(- 1.0f + 2.0f * ((float)(y + x * f->width) / (float)(f->height * f->width - 1)));
        }
    }
}

inline void createLeftToRightGradientFilter(Filter* f) {
    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            f->weights[y][x] = -1.0f + 2.0f * ((float)(x) / ((float)(f->width) - 1.0f));
        }
    }
}

inline void createRightToLeftGradientFilter(Filter* f) {
    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            f->weights[y][x] = -(- 1.0f + 2.0f * ((float)(x) / ((float)(f->width) - 1.0f)));
        }
    }
}

inline void createTopToBottomGradientFilter(Filter* f) {
    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            f->weights[y][x] = 2.0f * ((float)(y) / (float)(f->height - 1.0f)) - 1.0f;
        }
    }
}

inline void createBottomToTopGradientFilter(Filter* f) {
    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            f->weights[y][x] = -(2.0f * ((float)(y) / (float)(f->height - 1.0f))) + 1.0f;
        }
    }
}

inline void createTopLeftToBottomRightGradientFilter(Filter* f) {
    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            f->weights[y][x] = -1.0f + 2.0f * ((float)(x + y) / (float)(f->height + f->width - 2.0f));
        }
    }
}

inline void createBottomLeftToTopRightGradientFilter(Filter* f) {
    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            f->weights[y][x] = (float)(x) / ((float)(f->height) - 1.0f) - (float)(y) / ((float)(f->width) - 1.0f);
        }
    }
}

inline void createGaussianFilter(Filter* f) {

}

inline void createBalancedGaussianFilter(Filter* f) {

}

inline void createInverseGaussianFilter(Filter* f) {

}

inline void createModifiedGaussianFilter(Filter* f) {

}

inline void createConicalFilter(Filter* f) {

}
