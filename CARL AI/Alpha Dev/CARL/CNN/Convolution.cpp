 #include "Convolution.h"

Convolution::Convolution(int h, int w, FilterDimensions f) {
	input_w = w;
	input_h = h;
	stride = 0;
	k = new Kernel(f);
}

std::vector<std::vector<float>> Convolution::convolute(std::vector<std::vector<float>> input, int height, int width, int* output_h, int* output_w) {
	std::vector<std::vector<float>> convolved = std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f));
	
	*output_h = height - k->getRows();
	*output_w = width - k->getColumns();

	for (int i = 0; i < *output_h; i++) {
		std::vector<std::vector<float>> cached = std::vector<std::vector<float>>(k->getRows(), std::vector<float>(k->getColumns(), 0.0f));
		for (int j = 0; j < *output_w; j++) {
			for (int ki = 0; ki < k->getRows(); ki++) {
				for (int kj = 0; kj < k->getColumns(); kj++) {
					cached[ki][kj] = input[i + ki][j + kj];
				}
			}
			convolved[i][j] = k->getMax(cached);
		}
	}

	return convolved;
}