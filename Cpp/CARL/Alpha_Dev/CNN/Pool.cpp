#include "Pool.h"
#include "Filters.h"


static inline void maxPool(fmatrix* input, int stride) {
    fmatrix data = *input;
    int input_h = data.size();
    int input_w = data[0].size();
    fmatrix cache = fmatrix((input_h + 1) / stride, fvector((input_w + 1) / stride, 0.0f));

    for (int y = 0; y < input_h; y += stride) {
        for (int x = 0; x < input_w; x += stride) {
            float max = -1.0f;
            for (int ys = y; ys < y + stride && ys < (input_h); ++ys) {
                for (int xs = x; xs < x + stride && xs < (input_w); ++xs) {
                    float value = data[ys][xs];
                    if (max < value) { max = value; }
                }
            }
            cache[y / stride][x / stride] = max;
        }
        
    }

    *input = cache;
}

static inline void avgPool(fmatrix* input, int stride) {
    fmatrix data = *input;
    int input_h = data.size();
    int input_w = data[0].size();
    fmatrix cache = fmatrix((input_h + 1) / stride, fvector((input_w + 1) / stride, 0.0f));

    for (int y = 0; y < input_h; y += stride) {
        for (int x = 0; x < input_w; x += stride) {
            float avg = 0.0f;
            int avg_counter = 0;
            for (int ys = y; ys < y + stride && ys < (input_h); ++ys) {
                for (int xs = x; xs < x + stride && xs < (input_w); ++xs) {
                    avg += data[ys][xs];
                    avg_counter++;
                }
            }
            cache[y / stride][x / stride] = avg / avg_counter;
        }

    }

    *input = cache;
}

static inline void l2Pool(fmatrix* input, int stride) {
    fmatrix data = *input;
    fmatrix cache = fmatrix(data.size() / stride + 1, fvector(data[0].size() / stride + 1, 0.0f));

    for (int y = 0; y < input->size(); y += stride) {
        for (int x = 0; x < input[0].size(); x += stride) {
            float l2 = 0.0f;
            for (int ys = 0; ys < stride; ys++) {
                for (int xs = 0; xs < stride; xs++) {
                    float value = data[y + ys][x + xs];
                    l2 += (value * value);
                }
            }
            cache[y][x] = sqrtf(l2);
        }

    }

    *input = cache;
}

static inline void maxGlobalPool(fmatrix* input, int stride) {

}

static inline void avgGlobalPool(fmatrix* input, int stride) {

}

static inline void stochasticPool(fmatrix* input, int stride) {

}

static inline void roiPool(fmatrix* input, int stride) {

}

static inline void adaptivePool(fmatrix* input, int stride) {

}


inline void (*runPoolingFunction[])(fmatrix*, int) = {
    maxPool, avgPool, l2Pool, maxGlobalPool, avgGlobalPool, stochasticPool, roiPool, adaptivePool
};


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

fmatrix Pool::poolingFunction(fmatrix input)
{
    fmatrix sample = input;
    runPoolingFunction[pooling_style](&sample, stride);
    return sample;
}
