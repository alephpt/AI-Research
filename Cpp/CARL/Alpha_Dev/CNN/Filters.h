#pragma once
#include "../Types/Types.h"
#include <map>
#include <variant>

typedef struct Filter {
    fmatrix weights = fmatrix(0, fvector(0, 0.0f));
    int width = 0;
    int height = 0;
} Filter;

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
        NON_DISCRIMINATORY_FILTER,
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
    "Non-Discriminatory",
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

extern inline void oneXone(int* r, int* c);
extern inline void twoXtwo(int* r, int* c);
extern inline void threeXthree(int* r, int* c);
extern inline void fiveXfive(int* r, int* c);
extern inline void sevenXseven(int* r, int* c);
extern inline void elevenXeleven(int* r, int* c);
extern inline void oneXn(int* r, int* c, int n);
extern inline void twoXn(int* r, int* c, int n);
extern inline void threeXn(int* r, int* c, int n);
extern inline void nXone(int* r, int* c, int n);
extern inline void nXtwo(int* r, int* c, int n);
extern inline void nXthree(int* r, int* c, int n);
extern inline void createNonDiscriminatoryFilter(Filter* f);
extern inline void createRightEdgeFilter(Filter* f);
extern inline void createLeftEdgeFilter(Filter* f);
extern inline void createTopEdgeFilter(Filter* f);
extern inline void createBottomEdgeFilter(Filter* f);
extern inline void createTopRightCornerFilter(Filter* f);
extern inline void createBottomRightCornerFilter(Filter* f);
extern inline void createBottomLeftCornerFilter(Filter* f);
extern inline void createTopLeftCornerFilter(Filter* f);
extern inline void createAscendingFilter(Filter* f);
extern inline void createDescendingFilter(Filter* f);
extern inline void createVerticalAscendingFilter(Filter* f);
extern inline void createVerticalDescendingFilter(Filter* f);
extern inline void createLeftToRightGradientFilter(Filter* f);
extern inline void createRightToLeftGradientFilter(Filter* f);
extern inline void createTopToBottomGradientFilter(Filter* f);
extern inline void createBottomToTopGradientFilter(Filter* f);
extern inline void createTopLeftToBottomRightGradientFilter(Filter* f);
extern inline void createBottomLeftToTopRightGradientFilter(Filter* f);
extern inline void createGaussianFilter(Filter* f);
extern inline void createBalancedGaussianFilter(Filter* f);
extern inline void createInverseGaussianFilter(Filter* f);
extern inline void createModifiedGaussianFilter(Filter* f);
extern inline void createConicalFilter(Filter* f);
extern inline void createInverseConicalFilter(Filter* f);
extern void (*setFixedFilter[])(int* r, int* c);
extern void (*setDynamicFilter[])(int* r, int* c, int n);
extern void (*populateFilterStyle[])(Filter* f);