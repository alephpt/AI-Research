#pragma once
#include "../Types/Types.h"

typedef struct GANFeature {
	fscalar value;			// to track actual value						- actual value read, generated, or determined
	fscalar range;			// to determine limits							- used to narrow scope based on quality and deviation
	fscalar deviation;		// to determine offset of the value in a range	- used to adjust the range of target values
	fscalar quality;			// to determine the integrity					- target goal of 1
	int index;
} GANFeature;

// extract(data)
// normalize()				// normalize the values so they are in the same range
// standardize()			// standardize the values to have a mean of zero and a standard deviation of one
// transform()				// apply specific transformations to the values - logarithmic functions
// select()					// select a subset of features based on criteria, importance or relevance
// combine()				// used to combine multiple features using dot product or concatenation
// split()					// used to split a single feature into multiple features
// get_dimensions()			// used to get the dimensionality of the feature members
// save()					// used to store in 'memory'
// load()					// used to load from 'memory'