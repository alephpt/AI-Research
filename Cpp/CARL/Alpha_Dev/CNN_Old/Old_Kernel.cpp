
/*
static void (*lookupFilter[])(int* r, int* c) = {
    oneXone, twoXtwo, oneXthree, invalidFilter, threeXone, invalidFilter, threeXthree, fiveXfive, sevenXseven, elevenXeleven
};

static void (*lookupNFilter[])(int* r, int* c, int n) = {
    invalidNFilter, invalidNFilter, oneXn, invalidNFilter, nXone, invalidNFilter, invalidNFilter, invalidNFilter, invalidNFilter
};

static void (*lookupFilterStyle[])(Filter* filter) = {
    populateAscendingFilter,
    populateNegativeAscendingFilter,
    populateGradientFilter,
    populateVerticalGradientFilter,
    populateInverseGradientFilter,
    populateInverseVerticalGradientFilter,
    populateTLBRGradientFilter,
    populateBLTRGradientFilter,
    populateGaussianFilter,
    populateBalancedGaussianFilter,
    populateNegativeGaussianFilter,
    populateModifiedGaussianFilter,
    populateConicalFilter
};


    // Constructor / Destructor
Kernel::Kernel() {
    activation_type = RELU;
    style = GRADIENT_FILTER;
    setFilterDimensions(THREExTHREE);
}

Kernel::Kernel(Activation a) {
    activation_type = a;
    style = GRADIENT_FILTER;
    setFilterDimensions(THREExTHREE);
}

Kernel::Kernel(FilterStyle fs) {
    activation_type = RELU;
    style = fs;
    setFilterDimensions(THREExTHREE);
}

Kernel::Kernel(FilterDimensions fd) {
    activation_type = RELU;
    style = GRADIENT_FILTER;
    setFilterDimensions(fd);
}

Kernel::Kernel(Activation a, FilterStyle fs) {
    activation_type = a;
    style = fs;
    setFilterDimensions(THREExTHREE);
}

Kernel::Kernel(FilterDimensions fd, FilterStyle fs) {
    activation_type = RELU;
    style = fs;
    setFilterDimensions(fd);
}

Kernel::Kernel(Activation a, FilterDimensions fd) {
    activation_type = a;
    style = GRADIENT_FILTER;
    setFilterDimensions(fd);
}

Kernel::Kernel(Activation a, FilterDimensions fd, FilterStyle fs) {
    activation_type = a;
    style = fs;
    setFilterDimensions(fd);
}

Kernel::Kernel(FilterDimensions fd, int n) {
    activation_type = RELU;
    style = GRADIENT_FILTER;
    setFilterDimensions(fd, n);
}

Kernel::Kernel(FilterDimensions fd, int n, FilterStyle fs) {
    activation_type = RELU;
    style = fs;
    setFilterDimensions(fd, n);
}

Kernel::Kernel(Activation a, FilterDimensions fd, int n) {
    activation_type = a;
    style = GRADIENT_FILTER;
    setFilterDimensions(fd, n);
}

Kernel::Kernel(Activation a, FilterDimensions fd, int n, FilterStyle fs) {
    activation_type = a;
    style = fs;
    setFilterDimensions(fd, n);
}

Kernel::~Kernel() { filter.weights.clear(); }

    // public functions
void Kernel::setFilterDimensions(FilterDimensions f) { 
    dimensions = f; 
    lookupFilter[dimensions](&filter.rows, &filter.columns); 
    filter.weights = fmatrix(filter.rows, vector<fscalar>(filter.columns, 0.0f));
    populateFilter(style); 
}

void Kernel::setFilterDimensions(FilterDimensions f, int n) { 
    dimensions = f; 
    lookupNFilter[dimensions](&filter.rows, &filter.columns, n); 
    filter.weights = fmatrix(filter.rows, vector<fscalar>(filter.columns, 0.0f));
    populateFilter(style); 
}

void Kernel::populateFilter(FilterStyle s) {
    style = s;
    lookupFilterStyle[style](&filter);
    return;
}

void Kernel::adjustDimensions(FilterDimensions f)
{
    filter.weights.clear();
    setFilterDimensions(f);
}

void Kernel::adjustDimensions(FilterDimensions f, int n)
{
    filter.weights.clear();
    setFilterDimensions(f, n);
}

void Kernel::setStride(int s) { stride = s; }
void Kernel::setActivationType(Activation a) { activation_type = a; }
void Kernel::setFilterStyle(FilterStyle s) { style = s; populateFilter(style); }
fmatrix Kernel::getWeights() { return filter.weights; }
FilterDimensions Kernel::getDimensions() { return dimensions; }
Activation Kernel::getActivationType() { return activation_type; }
FilterStyle Kernel::getFilterStyle() { return style; }
int Kernel::getColumns() { return filter.columns; }
int Kernel::getRows() { return filter.rows; }


fscalar Kernel::getProductSum(fmatrix input) {
    size_t rows = input.size();
    size_t cols = input[0].size();
    fscalar sum = 0.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            sum += input[y][x] * filter.weights[y][x];
        }
    }

    return sum;
}

fscalar Kernel::getMax(fmatrix input)
{
    size_t rows = input.size();
    size_t cols = input[0].size();
    fscalar max = 0.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            fscalar newMax = input[y][x];
            
            if (newMax > max) { max = newMax; }
        }
    }

    return max;
}



fscalar Kernel::getMaxMean(fmatrix input)
{
    size_t rows = input.size();
    size_t cols = input[0].size();
    fscalar max = 0.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            fscalar maxMean = input[y][x] + filter.weights[y][x] / 2;
            if (max < maxMean) { max = maxMean; }
        }
    }

    return max;
}



fscalar Kernel::getSum(fmatrix input)
{
    size_t rows = input.size();
    size_t cols = input[0].size();
    fscalar sum = 0.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            sum += input[y][x] + filter.weights[y][x];
        }
    }

    return sum;
}

fscalar Kernel::getSumMean(fmatrix input)
{
    size_t rows = input.size();
    size_t cols = input[0].size();
    fscalar sum = 0.0f;
    size_t total = rows * cols;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            sum += input[y][x] + filter.weights[y][x];
        }
    }

    return sum / total;
}

fscalar Kernel::getMeanSum(fmatrix input)
{
    size_t rows = input.size();
    size_t cols = input[0].size();
    fscalar sum = 0.0f;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            sum += (input[y][x] + filter.weights[y][x]) / 2;
        }
    }

    return sum;
}

fscalar Kernel::getMean(fmatrix input)
{
    size_t rows = input.size();
    size_t cols = input[0].size();
    fscalar sum = 0.0f;
    size_t total = rows + cols;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            sum += (input[y][x] + filter.weights[y][x]) / 2;
        }
    }

    return sum / total;
}

void Kernel::printFilter()
{
    printf("%s %s Kernel\n", filterString[dimensions].c_str(), filterStyleString[style].c_str());
    for (int y = 0; y < filter.rows; y++) {
        printf("\t\t");
        for (int x = 0; x < filter.columns; x++) {
            printColor(filter.weights[y][x]);
        }
        printf("\n");
    }
    printf("\n");
    return;
}

*/