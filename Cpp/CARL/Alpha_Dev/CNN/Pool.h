#pragma once
#include "Filters.h"

const std::string poolingStyleString[] = {
    "Max Pooling",
    "Average Pooling",
    "L2 Pooling",
    "Stochastic Pooling",
    "Global Max Pooling",
    "Global Average Pooling"
};

typedef enum {
    MAX_POOLING,                // max value of a pool
    AVG_POOLING,                // average of a pool
    L2_POOLING,                 // root of sum of squares
    STOCHASTIC_POOLING,          // selects a random value
    GLOBAL_MAX_POOLING,         // finds maximum of all the values
    GLOBAL_AVG_POOLING         // computes average over all values
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

    fmatrix poolingFunction(fmatrix);
private:
    Filter* filter = new Filter;                     // matrix of values between -1.0 and 1.0
    FilterDimensions dimensions = TWOxTWO;        // store the constant e.g THREExTHREE
    FilterStyle filter_style = NON_DISCRIMINATORY_FILTER;           // used to determine the filter characteristics
    PoolingStyle pooling_style = MAX_POOLING;         // determines pooling implementation
    int stride = 2;

    void populateFilter();
};



