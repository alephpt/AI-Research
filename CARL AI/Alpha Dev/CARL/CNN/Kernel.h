#pragma once
#include "../Types/Types.h"
#include "Filters.h"
#include <map>
#include <variant>

typedef enum DynamicFilterDimensions {
        ONExN,
        TWOxN,
        THREExN,
        NxTHREE,
        NxTWO,
        NxONE,
} DynamicFilterDimensions;

typedef enum FixedFilterDimensions {
        ONExONE,
        TWOxTWO,
        THREExTHREE,
        FIVExFIVE,
        SEVENxSEVEN,
        ELEVENxELEVEN
} FixedFilterDimensions;

const std::map<std::variant<DynamicFilterDimensions, FixedFilterDimensions>, std::string> filterString = { 
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

static void (*lookupFixedFilter[])(int* r, int* c) = {
    oneXone, twoXtwo, threeXthree, fiveXfive, sevenXseven, elevenXeleven
};

static void (*lookupNFilter[])(int* r, int* c, int n) = {
    oneXn, twoXn, threeXn, nXthree, nXtwo, nXone
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
        NEGATIVE_ASCENDING_FILTER,
        GRADIENT_FILTER,
        VERTICAL_GRADIENT_FILTER,
        INVERSE_GRADIENT_FILTER,
        VERTICAL_INVERSE_GRADIENT_FILTER,
        TOP_LEFT_GRADIENT_FILTER,
        BOTTOM_LEFT_GRADIENT_FILTER,
        GAUSSIAN_FILTER,
        BALANCED_GAUSSIAN_FILTER,
        NEGATIVE_GAUSSIAN_FILTER,
        MODIFIED_GAUSSIAN_FILTER,
        CONICAL_FILTER
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
    "Negative Gradient", 
    "Gradient", 
    "Vertical Gradient", 
    "Inverse Gradient", 
    "Vertical Inverse Gradient", 
    "Top Left Gradient", 
    "Bottom_Left_Gradiant", 
    "Guassian", 
    "Balanced Gaussian",
    "Negative Guassian",
    "Modified Guassian",
    "Conical",
};

typedef struct Filter {
    fmatrix weights;
    int width;
    int height;
} Filter;

class Kernel {
public:
    Kernel();

    ~Kernel();

private:
    Filter filter;
};