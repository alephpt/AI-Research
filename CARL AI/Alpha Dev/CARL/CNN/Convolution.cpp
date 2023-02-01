 #include "Convolution.h"

Convolution::Convolution(int h, int w) {
	input_w = w;
	input_h = h;
	stride = 0;
	k = new Kernel();
}

Convolution::Convolution(Activation a, int h, int w) {
	input_w = w;
	input_h = h;
	stride = 0;
	k = new Kernel(a);
}

Convolution::Convolution(int h, int w, FilterDimensions f) {
	input_w = w;
	input_h = h;
	stride = 0;
	k = new Kernel(f);
}

Convolution::Convolution(Activation a, int h, int w, FilterDimensions f) {
	input_w = w;
	input_h = h;
	stride = 0;
	k = new Kernel(a, f);
}

std::vector<std::vector<float>> Convolution::convolute(std::vector<std::vector<float>> input, int height, int width, int* output_h, int* output_w) {
	std::vector<std::vector<float>> convolved = std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f));
	
	*output_h = height - k->getRows() + 1;
	*output_w = width - k->getColumns() + 1;

	for (int i = 0; i < *output_h; i++) {
		std::vector<std::vector<float>> cached = std::vector<std::vector<float>>(k->getRows(), std::vector<float>(k->getColumns(), 0.0f));
		for (int j = 0; j < *output_w; j++) {
			for (int ki = 0; ki < k->getRows(); ki++) {
				for (int kj = 0; kj < k->getColumns(); kj++) {
					cached[ki][kj] = input[i + ki][j + kj];
				}
			}

//			convolved[i][j] = activation(k->getActivationType(), k->getMax(cached));
			//convolved[i][j] = activation(k->getActivationType(), k->getMean(cached));
			convolved[i][j] = activation(k->getActivationType(), k->getProductSum(cached));
		}
	}

	return convolved;
}

std::vector<std::vector<float>> Convolution::paddedConvolute(std::vector<std::vector<float>> input, int height, int width, int* output_h, int* output_w) {
	std::vector<std::vector<float>> convolved = std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f));
	
	*output_h = height;
	*output_w = width;
	const int padding_y = k->getRows() / 2;
	const int padding_x = width - k->getColumns() / 2;

	for (int i = 0; i < height; i++) {
		std::vector<std::vector<float>> cached = std::vector<std::vector<float>>(k->getRows(), std::vector<float>(k->getColumns(), 0.0f));
		for (int j = 0; j < width; j++) {
			for (int ki = 0; ki < k->getRows(); ki++) {
				for (int kj = 0; kj < k->getColumns(); kj++) {
					if (i + ki < padding_y || j + kj < padding_x || 
						i + ki > height + padding_y || j + kj + padding_x) {
							cached[ki][kj] = 0.5f;
					} else {
						cached[ki][kj] = input[i + ki - padding_y][j + kj - padding_x];
					}
				}
			}

//			convolved[i][j] = activation(k->getActivationType(), k->getMax(cached));
			convolved[i][j] = activation(k->getActivationType(), k->getMean(cached));
			//convolved[i][j] = k->getProductSum(cached);
		}
	}

	return convolved;
}

std::vector<std::vector<float>> Convolution::dilationConvolute(std::vector<std::vector<float>> input, int height, int width, int* output_h, int* output_w) {
	std::vector<std::vector<float>> convolved = std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f));
	
	*output_h = height - k->getRows() + 1;
	*output_w = width - k->getColumns() + 1;

	for (int i = 0; i < *output_h; i++) {
		std::vector<std::vector<float>> cached = std::vector<std::vector<float>>(k->getRows(), std::vector<float>(k->getColumns(), 0.0f));
		for (int j = 0; j < *output_w; j++) {
			for (int ki = 0; ki < k->getRows(); ki += 2) {
				for (int kj = 0; kj < k->getColumns(); kj += 2) {
					cached[ki][kj] = input[i + ki][j + kj];
				}
			}

//			convolved[i][j] = activation(k->getActivationType(), k->getMax(cached));
			//convolved[i][j] = activation(k->getActivationType(), k->getMean(cached));
			convolved[i][j] = k->getProductSum(cached);
		}
	}

	return convolved;
}