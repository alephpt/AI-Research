
/*
CNNLayer::CNNLayer(int h, int w) {
	stride = 1;
	k = new Kernel();
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

CNNLayer::CNNLayer(int h, int w, int s) {
	stride = s + 1;
	k = new Kernel();
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

CNNLayer::CNNLayer(Activation a, int h, int w) {
	stride = 1;
	k = new Kernel(a);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

CNNLayer::CNNLayer(Activation a, int h, int w, int s) {
	stride = s + 1;
	k = new Kernel(a);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

CNNLayer::CNNLayer(int h, int w, FilterDimensions f) {
	stride = 1;
	k = new Kernel(f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

CNNLayer::CNNLayer(int h, int w, int s, FilterDimensions f) {
	stride = s + 1;
	k = new Kernel(f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

CNNLayer::CNNLayer(int h, int w, FilterDimensions f, int n) {
	stride = 1;
	k = new Kernel(f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

CNNLayer::CNNLayer(int h, int w, int s, FilterDimensions f, int n) {
	stride = s + 1;
	k = new Kernel(f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

CNNLayer::CNNLayer(Activation a, int h, int w, FilterDimensions f) {
	stride = 1;
	k = new Kernel(a, f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

CNNLayer::CNNLayer(Activation a, int h, int w, int s, FilterDimensions f) {
	stride = s + 1;
	k = new Kernel(a, f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

CNNLayer::CNNLayer(Activation a, int h, int w, FilterDimensions f, int n) {
	stride = 1;
	k = new Kernel(a, f, n);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

CNNLayer::CNNLayer(Activation a, int h, int w, int s, FilterDimensions f, int n) {
	stride = s + 1;
	k = new Kernel(a, f, n);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
	createNewSample();
}

void CNNLayer::setStride(int s) { stride = (s > k->getColumns()) ? k->getColumns() : (s > 0) ? s : 1; }

CNNFeature* CNNLayer::createNewFeature() {
	CNNFeature* newFeature = new CNNFeature;
	CNNSample* currentSample = getCurrentSample();

	if (currentSample->features.size() != 0) { currentSample->n_features++; }
	currentSample->features.push_back(newFeature);

	return newFeature;
}

CNNFeature* CNNLayer::getCurrentFeature()
{
	CNNSample* currentSample = getCurrentSample();

	if (currentSample->features.size() == 0) { return createNewFeature(); }

	return currentSample->features[currentSample->n_features];
}

CNNSample* CNNLayer::createNewSample() {
	CNNSample* newSample = new CNNSample;

	if (data->layers.size() != 0) { data->n_layers++; }
	data->layers.push_back(newSample);
	
	return newSample;
}

CNNSample* CNNLayer::getCurrentSample()
{
	if (data->layers.size() == 0) { return createNewSample(); }

	return data->layers[data->n_layers];
}



void (*ConvolutionFn[])(CNNFeature*, CNNFeature*, Kernel*, int) = {
	validConvolution, paddedConvolution, dilationConvolution, paddedDilationConvolution
};

void CNNLayer::convolute(ConvolutionType convolution_type, CNNFeature* input) {
	CNNSample* sample = getCurrentSample();
	CNNFeature* newFeature = createNewFeature();
	ConvolutionFn[convolution_type](newFeature, input, k, stride);
	newFeature->filter_style = k->getFilterStyle();
	newFeature->filter = k->getWeights();
	sample->features.push_back(newFeature);
	newFeature->index = sample->n_features++;


	if (sample->kernel_dimensions != k->getDimensions()) {
		sample->kernel_dimensions = k->getDimensions();
	}

	if (sample->activation_type != k->getActivationType()) {
		sample->activation_type = k->getActivationType();
	}
}

void CNNLayer::printCurrentFeature()
{
	CNNFeature* feature = getCurrentFeature();

	printf("\noutput vector - %d x %d\n", feature->height, feature->width);
	
	for (int i = 0; i < feature->values[i].size(); i++) {
		printFMatrix(feature->values[i]);
	}
}
*/