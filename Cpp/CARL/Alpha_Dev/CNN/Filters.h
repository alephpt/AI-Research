#pragma once
#include "../Types/Types.h"
#include <cmath>
#include <map>
#include <variant>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

typedef enum {
        ONExN,
        TWOxN,
        THREExN,
        NxTHREE,
        NxTWO,
        NxONE,
} DynamicFilterDimensions;

typedef enum  {
        ONExONE,
        TWOxTWO,
        THREExTHREE,
        FIVExFIVE,
        SEVENxSEVEN,
        ELEVENxELEVEN
} FixedFilterDimensions;

typedef std::variant<DynamicFilterDimensions, FixedFilterDimensions> FilterDimensions;

const std::map<FilterDimensions, std::string> filterDimensionsString = { 
    {ONExONE, "ONExONE"}, 
    {TWOxTWO, "TWOxTWO"},
    {THREExTHREE, "THREExTHREE"}, 
    {ONExN, "ONExN"}, 
    {TWOxN, "TWOxN"},
    {THREExN, "THREExN"},
    {NxTHREE, "NxTHREE"},
    {NxTWO, "NxTWO"},
    {NxONE, "NxONE"}, 
    {FIVExFIVE, "FIVExFIVE"}, 
    {SEVENxSEVEN, "SEVENxSEVEN"}, 
    {ELEVENxELEVEN, "ELEVENxELEVEN"} 
};

typedef enum FilterStyle {
        RIGHT_EDGE_FILTER,
        LEFT_EDGE_FILTER,
        TOP_EDGE_FILTER,
        BOTTOM_EDGE_FILTER,
        TOP_RIGHT_CORNER_FILTER,
        BOTTOM_RIGHT_CORNER_FILTER,
        BOTTOM_LEFT_CORNER_FILTER,
        TOP_LEFT_CORNER_FILTER,
        ASCENDING_FILTER,
        DESCENDING_FILTER,
        VERTICAL_ASCENDING_FILTER,
        VERTICAL_DESCENDING_FILTER,
        LtoR_GRADIENT_FILTER,
        RtoL_GRADIENT_FILTER,
        TtoB_GRADIENT_FILTER,
        BtoT_GRADIENT_FILTER,
        TLtoBR_GRADIENT_FILTER,
        BLtoTR_GRADIENT_FILTER,
        GAUSSIAN_FILTER,
        BALANCED_GAUSSIAN_FILTER,
        NEGATIVE_GAUSSIAN_FILTER,
        MODIFIED_GAUSSIAN_FILTER,
        CONICAL_FILTER,
        INVERSE_CONICAL_FILTER
} FilterStyle;

const std::string filterStyleString[] = {   
    "Right Edge",
    "Left Edge",
    "Top Edge",
    "Bottom Edge",
    "Top Right Corner",
    "Bottom Right Corner",
    "Bottom Left Corner",
    "Top Left Corner",
    "Ascending", 
    "Descending",
    "Vertical Ascending",
    "Vertical Descending",
    "Left-To-Right Gradient", 
    "Right-To-Left Gradient",
    "Top-To-Bottom Gradient", 
    "Bottom-To-Top Gradient", 
    "Top Left to Bottom Right Gradient", 
    "Bottom Left to Top Right Gradiant", 
    "Guassian", 
    "Balanced Gaussian",
    "Negative Guassian",
    "Modified Guassian",
    "Conical",
    "Inverse Conical",
};

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
    float y_center = (float)f->height / 2.0f;
    float x_center = (float)f->width / 2.0f;
    float y_sigma = y_center / 3.0f;
    float x_sigma = y_center / 3.0f;

    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            float dx = (float)x + 0.5f - x_center;
            float dy = (float)y + 0.5f - y_center;
            float radial = sqrtf(dx * dx + dy * dy);
            f->weights[y][x] = (float)expf(-radial * radial / (2 * x_sigma * y_sigma));
        }
    }
}

inline void createBalancedGaussianFilter(Filter* f) {
    float y_center = (float)f->height / 2.0f;
    float x_center = (float)f->width / 2.0f;
    float y_sigma = y_center / 2.0f;
    float x_sigma = y_center / 2.0f;

    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            float dx = (float)x + 0.5f - x_center;
            float dy = (float)y + 0.5f - y_center;
            float radial = sqrtf(dx * dx + dy * dy);
            f->weights[y][x] = 2.0f * ((float)expf(-radial * radial / (2 * x_sigma * y_sigma)) - 0.5f);
        }
    }
}

inline void createInverseGaussianFilter(Filter* f) {
    float y_center = (float)f->height / 2.0f;
    float x_center = (float)f->width / 2.0f;
    float y_sigma = y_center / 2.0f;
    float x_sigma = y_center / 2.0f;

    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            float dx = (float)x + 0.5f - x_center;
            float dy = (float)y + 0.5f - y_center;
            float radial = sqrtf(dx * dx + dy * dy);
            f->weights[y][x] = 2.0f * (-(float)expf(-radial * radial / (2 * x_sigma * y_sigma)) + 0.5f);
        }
    }
}

inline void createModifiedGaussianFilter(Filter* f) {
    float y_center = (float)f->height / 2.0f;
    float x_center = (float)f->width / 2.0f;
    float y_sigma = y_center / 2.0f;
    float x_sigma = y_center / 2.0f;

    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            float dx = (float)x + 0.5f - x_center;
            float dy = (float)y + 0.5f - y_center;
            float radial = sqrtf(dx * dx + dy * dy);
            f->weights[y][x] = 1.55f * ((float)expf(-radial * radial / (2 * x_sigma * y_sigma)) - 0.366f);
        }
    }
}

inline void createConicalFilter(Filter* f) {
    float y_center = (float)f->height / 2.0f;
    float x_center = (float)f->width / 2.0f;

    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            float theta = std::atan2f(y - y_center, x - x_center);
            theta = (theta >= 0) ? theta : 2.0f * (float)M_PI + theta;
            f->weights[y][x] = (2.0f * theta / (2.0f * (float)M_PI) - 1.0f);
        }
    }
}

inline void createInverseConicalFilter(Filter* f) {
    float y_center = (float)f->height / 2.0f;
    float x_center = (float)f->width / 2.0f;

    for (int y = 0; y < f->height; y++) {
        for (int x = 0; x < f->width; x++) {
            float theta = std::atan2f(y - y_center, x - x_center);
            theta = (theta >= 0) ? theta : 2.0f * (float)M_PI + theta;
            f->weights[y][x] = (2.0f * theta / (2.0f * (float)M_PI) - 1.0f) * -1.0f;
        }
    }
}

