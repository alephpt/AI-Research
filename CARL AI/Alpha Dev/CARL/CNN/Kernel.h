#pragma once
#include <vector>
#include <string>

const std::string filterString[] = { "ONExONE", "ONExTHREE", "ONExN", "THREExONE", "NxONE", "THREExTHREE", "FIVExFIVE", "SEVENxSEVEN", "ELEVENxELEVEN" };

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


typedef struct Filter {
    std::vector<std::vector<float>> values;
    int rows, columns;
} Filter;


class Kernel {
    public:
        Kernel(FilterDimensions, int);
        Kernel(FilterDimensions);
        ~Kernel();

        void print();
        void adjustDimensions(FilterDimensions);
        void adjustDimensions(FilterDimensions, int);
        int getRows();
        int getColumns();
        float getMax(std::vector<std::vector<float>>);
        float getMaxMean(std::vector<std::vector<float>>);
        float getSum(std::vector<std::vector<float>>);
        float getSumMean(std::vector<std::vector<float>>);
        float getMean(std::vector<std::vector<float>>);
        float getMeanSum(std::vector<std::vector<float>>);


    private:
        Filter filter;
        FilterDimensions dims;
};