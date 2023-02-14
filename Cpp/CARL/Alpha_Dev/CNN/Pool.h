#pragma once
#include "Filters.h"

const std::string poolingStyleString[] = {
    "Max Pooling",
    "Average Pooling",
    "L2 Pooling",
    "Global Max Pooling",
    "Global Average Pooling",
    "Stochastic Pooling",
    "Region of Interest Pooling",
    "Adaptive Pooling"
};

typedef enum {
    MAX_POOLING,                // max value of a pool
    AVG_POOLING,                // average of a pool
    L2_POOLING,                 // root of sum of squares
    MAX_GLOBAL_POOLING,         // finds maximum of all the values
    AVG_GLOBAL_POOLING,         // computes average over all values
    STOCHASTIC_POOLING,         // selects a random value
    ROI_POOLING,                // used to pool over regions of interest
    ADAPTIVE_POOLING            // dynamically computes the pooling size
} PoolingStyle;

class Pool{
public:
    Pool();
    Pool(PoolingStyle style);
    Pool(FilterStyle);
    Pool(FixedFilterDimensions);
    Pool(DynamicFilterDimensions, int);
    Pool(PoolingStyle, FixedFilterDimensions);
    Pool(PoolingStyle, DynamicFilterDimensions, int);
    Pool(PoolingStyle, FilterStyle);
    Pool(FixedFilterDimensions, FilterStyle);
    Pool(DynamicFilterDimensions, int, FilterStyle);
    Pool(PoolingStyle, FixedFilterDimensions, FilterStyle);
    Pool(PoolingStyle, DynamicFilterDimensions, int, FilterStyle);
    ~Pool();

    int getStride();
    int getFilterWidth();
    int getFilterHeight();
    fmatrix getFilterWeights();
    Filter* getFilter();
    FilterStyle getFilterStyle();
    std::string getFilterStyleString();
    FilterDimensions getFilterDimensions();
    std::string getFilterDimensionsString();
    PoolingStyle getPoolingStyle();
    std::string getPoolingStyleString();

    void setStride(int);
    void setFilterStyle(FilterStyle);
    void setFilterDimensions(FixedFilterDimensions);
    void setFilterDimensions(DynamicFilterDimensions, int);
    void setFilterParameters(FixedFilterDimensions, FilterStyle);
    void setFilterParameters(DynamicFilterDimensions, int, FilterStyle);

private:
    Filter* filter = new Filter;                     // matrix of values between -1.0 and 1.0
    FilterDimensions dimensions = TWOxTWO;        // store the constant e.g THREExTHREE
    FilterStyle filter_style = NON_DISCRIMINATORY_FILTER;           // used to determine the filter characteristics
    PoolingStyle pooling_style = MAX_POOLING;         // determines pooling implementation
    int stride = 2;

    void populateFilter();
};