#pragma once
#include "Filters.h"
#include "../Types/Types.h"

class Kernel {
public:
    Kernel();
    Kernel(FilterStyle);
    Kernel(FixedFilterDimensions);
    Kernel(FixedFilterDimensions, FilterStyle);
    Kernel(DynamicFilterDimensions, int);
    Kernel(DynamicFilterDimensions, int, FilterStyle);
    ~Kernel();

    void printFilter();
    
    float getBias();
    FilterStyle getFilterStyle();
    std::string getFilterStyleString();
    FilterDimensions getFilterDimensions();
    std::string getFilterDimensionsString();
    Filter* getFilter();
    int getFilterWidth();
    int getFilterHeight();
    fmatrix getFilterWeights();

    void setBias(float);
    void setFilterStyle(FilterStyle);
    void setFilterDimensions(FixedFilterDimensions);
    void setFilterDimensions(DynamicFilterDimensions, int);
    void setFilterParameters(FixedFilterDimensions, FilterStyle);
    void setFilterParameters(DynamicFilterDimensions, int, FilterStyle);
private:
    Filter* filter;
    FilterDimensions filter_dimensions;
    FilterStyle filter_style;
    float bias;

    void populateFilter();
};


