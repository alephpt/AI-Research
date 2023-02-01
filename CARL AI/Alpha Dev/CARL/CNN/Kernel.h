#pragma once
#include "../Types/Activation.h"
#include <vector>
#include <string>

const std::string filterString[] = { "ONExONE", "ONExTHREE", "ONExN", "THREExONE", "NxONE", "THREExTHREE", "FIVExFIVE", "SEVENxSEVEN", "ELEVENxELEVEN" };
const std::string filterStyleString[] = { "Ascending", "Offset", "Ascending Offset", "Vertical Offset", "Inverse Offset", "Vertical Inverse Offset", "Top Left Gradient", "Bottom_Left_Gradiant", "Guassian", "Negative Guassian", "Conical"};

typedef enum FilterDimensions {
        ONExONE,
        ONExTHREE,
        ONExN,
        THREExONE,
        NxONE,
        THREExTHREE,
        FIVExFIVE,
        SEVENxSEVEN,
        ELEVENxELEVEN
} FilterDimensions;

typedef enum FilterStyle {
        ASCENDING_FILTER,
        OFFSET_FILTER,
        ASCENDING_OFFSET_FILTER,
        VERTICAL_OFFSET_FILTER,
        INVERSE_OFFSET_FILTER,
        VERTICAL_INVERSE_OFFSET_FILTER,
        TOP_LEFT_GRADIENT_FILTER,
        BOTTOM_LEFT_GRADIENT_FILTER,
        GAUSSIAN_FILTER,
        NEGATIVE_GAUSSIAN_FILTER,
        CONICAL_FILTER
} FilterStyle;

typedef struct Filter {
    std::vector<std::vector<float>> weights;
    int rows, columns;
} Filter;


class Kernel {
    public:
        Kernel();
        Kernel(FilterDimensions);
        Kernel(Activation);
        Kernel(Activation, FilterDimensions);
        Kernel(FilterDimensions, int);
        Kernel(Activation, FilterDimensions, int);

        ~Kernel();

        void print();
        int getRows();
        int getColumns();
        Activation getActivationType();
        void setStride(int);
        void setFilterType(FilterStyle);
        void adjustDimensions(FilterDimensions);
        void adjustDimensions(FilterDimensions, int);
        float getMax(std::vector<std::vector<float>>);
        float getMaxMean(std::vector<std::vector<float>>);
        float getProductSum(std::vector<std::vector<float>>);
        float getSum(std::vector<std::vector<float>>);
        float getSumMean(std::vector<std::vector<float>>);
        float getMeanSum(std::vector<std::vector<float>>);
        float getMean(std::vector<std::vector<float>>);

    private:
        void populateFilter(FilterStyle);
        int stride = 0;
        Filter filter;
        FilterStyle style;
        FilterDimensions dims;
        Activation activation_type;
};