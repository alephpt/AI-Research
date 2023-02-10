#pragma once
#include "Filters.h"
#include "../Types/Types.h"
#include <map>
#include <variant>

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

class Kernel {
public:
    Kernel();
    Kernel(FilterStyle);
    Kernel(FixedFilterDimensions);
    Kernel(FixedFilterDimensions, FilterStyle);
    Kernel(DynamicFilterDimensions, int);
    Kernel(DynamicFilterDimensions, int, FilterStyle);
    ~Kernel();

    void setFilterParameters(FixedFilterDimensions, FilterStyle);
    void setFilterParameters(DynamicFilterDimensions, int, FilterStyle);
    void setFilterDimensions(FixedFilterDimensions);
    void setFilterDimensions(DynamicFilterDimensions, int);
    void setFilterStyle(FilterStyle);
    void printFilter();
private:
    Filter* filter;
    FilterDimensions filter_dimensions;
    FilterStyle filter_style;

    void populateFilter();
};