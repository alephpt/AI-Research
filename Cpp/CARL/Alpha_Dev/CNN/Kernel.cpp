#include "Kernel.h"

Kernel::Kernel() 
{
    filter = new Filter();
    filter_style = NON_DISCRIMINATORY_FILTER;
    setFilterDimensions(THREExTHREE);
}

Kernel::Kernel(FilterStyle s)
{
    filter = new Filter();
    filter_style = s;
    setFilterDimensions(THREExTHREE);
}

Kernel::Kernel(FixedFilterDimensions d)
{
    filter = new Filter();
    filter_style = NON_DISCRIMINATORY_FILTER;
    setFilterDimensions(d);
}

Kernel::Kernel(FixedFilterDimensions d, FilterStyle s)
{
    filter = new Filter();
    filter_style = s;
    setFilterDimensions(d);
}

Kernel::Kernel(DynamicFilterDimensions d, int n)
{
    filter = new Filter();
    filter_style = NON_DISCRIMINATORY_FILTER;
    setFilterDimensions(d, n);
}

Kernel::Kernel(DynamicFilterDimensions d, int n, FilterStyle s)
{
    filter = new Filter();
    filter_style = s;
    setFilterDimensions(d, n);
}

Kernel::~Kernel() 
{
    filter->weights.clear();
}

void Kernel::setFilterParameters(FixedFilterDimensions d, FilterStyle s) {
    filter_style = s;
    setFilterDimensions(d);
}

void Kernel::setFilterParameters(DynamicFilterDimensions d, int n, FilterStyle s) {
    filter_style = s;
    setFilterDimensions(d, n);
}

void Kernel::setFilterDimensions(FixedFilterDimensions d)
{
    filter_dimensions = d;
    setFixedFilter[d](&filter->height, &filter->width);
    populateFilter();
}

void Kernel::setFilterDimensions(DynamicFilterDimensions d, int n)
{
    filter_dimensions = d;
    setDynamicFilter[d](&filter->height, &filter->width, n);
    populateFilter();
}

void Kernel::setFilterStyle(FilterStyle s) {
    filter_style = s;
    populateFilter();
}

Filter* Kernel::getFilter() { return filter; }
float Kernel::getBias() { return bias; }

FilterStyle Kernel::getFilterStyle() { return filter_style; }
std::string Kernel::getFilterStyleString() { return filterStyleString[filter_style]; }
FilterDimensions Kernel::getFilterDimensions() { return filter_dimensions; }
std::string Kernel::getFilterDimensionsString() { return filterDimensionsString.at(filter_dimensions); }
void Kernel::setBias(float b) { bias = b; }
int Kernel::getFilterWidth() { return filter->width; }
int Kernel::getFilterHeight() { return filter->height; }
fmatrix Kernel::getFilterWeights() { return filter->weights; }

void Kernel::populateFilter()
{
    filter->weights = fmatrix(filter->height, fvector(filter->width, 0.0f));
    
    if (filter->width < 2 && filter->height < 2) {
        filter->weights[0][0] = 1.0f;
        return;
    }

    populateFilterStyle[filter_style](filter);
}

void Kernel::printFilter() {
    printf("%s(%dx%d) %s Filter\n", filterDimensionsString.at(filter_dimensions).c_str(), filter->height, filter->width, filterStyleString[filter_style].c_str());
    printFMatrix(filter->weights);
    printf("\n\n");
}