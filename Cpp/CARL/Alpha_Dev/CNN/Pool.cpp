#include "Pool.h"
#include "Filters.h"

Pool::Pool() {
    filter = new Filter();
    pooling_style = MAX_POOLING;
    setFilterDimensions(TWOxTWO);
}

Pool::Pool(PoolingStyle style) {
    filter = new Filter();
    pooling_style = style;
    setFilterDimensions(TWOxTWO);
}

Pool::Pool(FilterStyle fs) {
    filter = new Filter();
    pooling_style = MAX_POOLING;
    setFilterParameters(TWOxTWO, fs);
}

Pool::Pool(FixedFilterDimensions fd) {
    filter = new Filter();
    pooling_style = MAX_POOLING;
    setFilterDimensions(fd);
}

Pool::Pool(DynamicFilterDimensions fd, int n) {
    filter = new Filter();
    pooling_style = MAX_POOLING;
    setFilterDimensions(fd, n);
}

Pool::Pool(PoolingStyle ps, FixedFilterDimensions fd) {
    filter = new Filter();
    pooling_style = ps;
    setFilterDimensions(fd);
}

Pool::Pool(PoolingStyle ps, DynamicFilterDimensions fd, int n) {
    filter = new Filter();
    pooling_style = ps;
    setFilterDimensions(fd, n);
}

Pool::Pool(PoolingStyle ps, FilterStyle fs) {
    filter = new Filter();
    pooling_style = ps;
    setFilterParameters(TWOxTWO, fs);
}

Pool::Pool(FixedFilterDimensions fd, FilterStyle fs) {
    filter = new Filter();
    pooling_style = MAX_POOLING;
    setFilterParameters(fd, fs);
}

Pool::Pool(DynamicFilterDimensions fd, int n, FilterStyle fs) {
    filter = new Filter();
    pooling_style = MAX_POOLING;
    setFilterParameters(fd, n, fs);
}

Pool::Pool(PoolingStyle ps, FixedFilterDimensions fd, FilterStyle fs) {
    filter = new Filter();
    pooling_style = ps;
    setFilterParameters(fd, fs);
}

Pool::Pool(PoolingStyle ps, DynamicFilterDimensions fd, int n, FilterStyle fs) {
    filter = new Filter();
    pooling_style = ps;
    setFilterParameters(fd, n, fs);
}

Pool::~Pool() {
    filter->weights.clear();
}


void Pool::populateFilter()
{
    filter->weights = fmatrix(filter->height, fvector(filter->width, 0.0f));

    if (filter->width < 2 && filter->height < 2) {
        filter->weights[0][0] = 1.0f;
        return;
    }

    populateFilterStyle[filter_style](filter);
}

FilterStyle Pool::getFilterStyle() { return filter_style; }
std::string Pool::getFilterStyleString() { return filterStyleString[filter_style]; }
FilterDimensions Pool::getFilterDimensions() { return dimensions; }
std::string Pool::getFilterDimensionsString() { return filterDimensionsString.at(dimensions); }
PoolingStyle Pool::getPoolingStyle() { return pooling_style; }
std::string Pool::getPoolingStyleString() { return poolingStyleString[pooling_style]; }
Filter* Pool::getFilter() { return filter; }
int Pool::getFilterHeight() { return filter->height; }
int Pool::getStride() { return stride; }
int Pool::getFilterWidth() { return filter->width; }
fmatrix Pool::getFilterWeights() { return filter->weights; }

void Pool::setStride(int s) { stride = s; }

void Pool::setFilterStyle(FilterStyle fs) {
    filter_style = fs; 
    populateFilter(); 
}

void Pool::setFilterDimensions(FixedFilterDimensions fd) { 
    dimensions = fd;
    setFixedFilter[fd](&filter->height, &filter->width);
    populateFilter();
}

void Pool::setFilterDimensions(DynamicFilterDimensions fd, int n) { 
    dimensions = fd;
    setDynamicFilter[fd](&filter->height, &filter->width, n); 
    populateFilter();
}

void Pool::setFilterParameters(FixedFilterDimensions fd, FilterStyle fs) { 
    filter_style = fs;
    dimensions = fd;
    setFixedFilter[fd](&filter->height, &filter->width);
    populateFilter();
}

void Pool::setFilterParameters(DynamicFilterDimensions fd, int n, FilterStyle fs) {
    filter_style = fs;
    dimensions = fd;
    setDynamicFilter[fd](&filter->height, &filter->width, n);
    populateFilter();
}