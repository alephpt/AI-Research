#pragma once
#include "Filters.h"

typedef enum {
    MAX_POOLING,                // max value of a pool
    AVG_POOLING,                // average of a pool
    L2_POOLING,                 // root of sum of squares
    MAX_GLOBAL_POOLING,         // finds maximum of all the values
    AVG_GLOBAL_POOLING,         // computes average over all values
    STOCHASTIC_POOLING,         // selects a random value
    ROL_POOLING,                // used to pool over regions of interest
    ADAPTIVE_POOLING            // dynamically computes the pooling size
} PoolingStyle;

typedef struct PoolingLayer{
    Filter* filter;
    FilterDimensions dimensions;
    PoolingStyle style;
} PoolingLayer;