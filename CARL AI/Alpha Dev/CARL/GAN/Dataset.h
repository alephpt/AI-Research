#pragma once
#include "Sample.h"

typedef struct Dataset {
	char* name;
	int n_samples;
	Sample* samples;
} Dataset;

// Dataset* createDataSet(int n_samples);
// void addSample(Dataset* dataset, Sample* sample)
// void shuffleDataset(Dataset* dataset);
// Dataset* splitDataset(Dataset* dataset, fscalar training_percentage);				// splits part of the original dataset, and returns the remaining part
// Dataset* mergeDataset(char* new_name Dataset* dataset_A, Dataset* dataset_ B);	// merges dataset_B into dataset_A, leaving both original datasets in tact
// void destroyData(Dataset* dataset);