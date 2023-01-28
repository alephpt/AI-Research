#pragma once
#include "Feature.h"
#include <cstdio>

typedef struct Sample {
    int n_features;
    char* family;
    Feature* features;
} Sample;

// Sample* initSample();
// void setFeature(Sample* sample, Feature* feature);
// Feature* getFeature(Sample* sample, int feature_index);
// char* getFamily(Sample* sample);
// addFeature(Sample* sample, Feature* feature);
// removeFeature(Sample* sample, char* family);
// saveSample(Sample* sample);
// loadSample(?);
// printSample(Sample* sample);
// destroySample(Sample* sample);