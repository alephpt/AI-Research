#pragma once
#include "CNNData.h"
#include "Kernel.h"

void convolute(CNNFeature* output, CNNFeature* input, Kernel* k, int stride) {
	output->height = (input->height - (k->getRows() / (stride + 1))) + 1;
	output->width = (input->width - (k->getColumns() / (stride + 1))) + 1;

	for (int i = 0; i < input->height; i += stride + 1) {
		std::vector<std::vector<float>> cached = std::vector<std::vector<float>>(k->getRows(), std::vector<float>(k->getColumns(), 0.0f));
		for (int j = 0; j < input->width; j += stride + 1) {
			for (int ki = 0; ki < k->getRows(); ki++) {
				for (int kj = 0; kj < k->getColumns(); kj++) {
					cached[ki][kj] = input->values[i + ki][j + kj];
				}
			}
			output->values[i][j] = activation(k->getActivationType(), k->getProductSum(cached));
		}
	}
}

void paddedConvolute(CNNFeature* output, CNNFeature* input, Kernel* k, int stride) {
	const int padding_y = k->getRows() / 2;
	const int padding_x = k->getColumns() / 2;
    output->height = (input->height + padding_y) / stride;
	output->width = (input->width + padding_x) / stride;

	for (int i = 0; i < input->height; i += stride + 1) {
		std::vector<std::vector<float>> cached = std::vector<std::vector<float>>(k->getRows(), std::vector<float>(k->getColumns(), 0.0f));
		for (int j = 0; j < input->width; j += stride + 1) {
			for (int ki = 0; ki < k->getRows(); ki++) {
				for (int kj = 0; kj < k->getColumns(); kj++) {
					if (i + ki < padding_y || j + kj < padding_x || 
						i + ki > input->height || j + kj > input->width) {
							cached[ki][kj] = 0.5f;
					} else {
						cached[ki][kj] = input->values[i + ki - padding_y][j + kj - padding_x];
					}
				}
			}

			output->values[i][j] = activation(k->getActivationType(), k->getProductSum(cached));
		}
	}
}

void dilationConvolute(CNNFeature* output, CNNFeature* input, Kernel* k, int stride) {
	output->height = (input->height - (k->getRows() + 1)) / stride;
	output->width = (input->width - (k->getColumns() + 1)) / stride;

	for (int i = 0; i < input->height; i += 1 + stride) {
		std::vector<std::vector<float>> cached = std::vector<std::vector<float>>(k->getRows(), std::vector<float>(k->getColumns(), 0.0f));
		for (int j = 0; j < input->width; j += 1 + stride) {
			for (int ki = 0; ki < k->getRows(); ki += 2) {
				for (int kj = 0; kj < k->getColumns(); kj += 2) {
					cached[ki][kj] = input->values[i + ki][j + kj];
				}
			}

			output->values[i][j] = activation(k->getActivationType(), k->getProductSum(cached));
		}
	}
}