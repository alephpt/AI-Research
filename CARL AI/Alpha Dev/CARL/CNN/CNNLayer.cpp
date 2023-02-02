#include "CNNLayer.h"
#include "CNNLayer_Helper.h"

CNNLayer::CNNLayer(int h, int w) {
	stride = 0;
	k = new Kernel();
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

CNNLayer::CNNLayer(int h, int w, int s) {
	stride = s;
	k = new Kernel();
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

CNNLayer::CNNLayer(Activation a, int h, int w) {
	stride = 0;
	k = new Kernel(a);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

CNNLayer::CNNLayer(Activation a, int h, int w, int s) {
	stride = s;
	k = new Kernel(a);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

CNNLayer::CNNLayer(int h, int w, FilterDimensions f) {
	stride = 0;
	k = new Kernel(f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

CNNLayer::CNNLayer(int h, int w, int s, FilterDimensions f) {
	stride = s;
	k = new Kernel(f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

CNNLayer::CNNLayer(int h, int w, FilterDimensions f, int n) {
	stride = 0;
	k = new Kernel(f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

CNNLayer::CNNLayer(int h, int w, int s, FilterDimensions f, int n) {
	stride = s;
	k = new Kernel(f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

CNNLayer::CNNLayer(Activation a, int h, int w, FilterDimensions f) {
	stride = 0;
	k = new Kernel(a, f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

CNNLayer::CNNLayer(Activation a, int h, int w, int s, FilterDimensions f) {
	stride = s;
	k = new Kernel(a, f);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

CNNLayer::CNNLayer(Activation a, int h, int w, FilterDimensions f, int n) {
	stride = 0;
	k = new Kernel(a, f, n);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

CNNLayer::CNNLayer(Activation a, int h, int w, int s, FilterDimensions f, int n) {
	stride = s;
	k = new Kernel(a, f, n);
	data = new CNNData;
	data->input.width = w;
	data->input.height = h;
}

void CNNLayer::setStride(int s) { stride = s; }

void (*ConvolutionFn[])(CNNFeature*, CNNFeature*, Kernel*, int) = {
	convolute, paddedConvolute, dilationConvolute
};

void CNNLayer::convolute(ConvolutionType convolution_type) {
	CNNFeature* newFeature = new CNNFeature;
	ConvolutionFn[convolution_type](newFeature, &data->input, k, stride);
	newFeature->filter_style = k->getFilterStyle();
	newFeature->filter = k->getWeights();
	newFeature->index = data->layers[data->n_layers].n_features++;
	data->layers[data->n_layers].features.push_back(newFeature);
	data->layers[data->n_layers].kernel_dimensions = k->getDimensions(); 		// need to add safety checks
	data->layers[data->n_layers].activation_type = k->getActivationType();  	// need to add safety checks
}