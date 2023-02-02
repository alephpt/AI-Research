#pragma once
#include "CNNData.h"
#include "Kernel.h"

void validConvolution(CNNFeature* output, CNNFeature* input, Kernel* k, int stride) {
	int kernel_rows = k->getRows();
	int kernel_cols = k->getColumns();
	output->height = (input->height - kernel_rows) / stride + 1;
	output->width = (input->width - kernel_cols) / stride + 1;
	output->values = matrix(output->height, std::vector<float>(output->width, 0.0f));
	matrix cached = matrix(kernel_rows, std::vector<float>(kernel_cols, 0.0f));

	for (int i = 0; i < output->height; ++i) {
		int i_step = i * stride;
		for (int j = 0; j < output->width; ++j)  {
			int j_step = j * stride;
			for (int ki = 0; ki < kernel_rows && (i_step + ki) < input->height; ki++) {
				for (int kj = 0; kj < kernel_cols && (j_step + kj) < input->width; kj++) {
					cached[ki][kj] = input->values[i_step + ki][j_step + kj];
				}
			}
			output->values[i][j] = activation(k->getActivationType(), k->getProductSum(cached));
		}
		cached = matrix(kernel_rows, std::vector<float>(kernel_cols, 0.0f));
	}
}

void paddedConvolution(CNNFeature* output, CNNFeature* input, Kernel* k, int stride) {
	int kernel_rows = k->getRows();
	int kernel_cols = k->getColumns();
	const int padding_y = kernel_rows / 2;
	const int padding_x = kernel_cols / 2;
    output->height = input->height / stride + 1;
	output->width = input->width / stride + 1;
	output->values = matrix(output->height, std::vector<float>(output->width, 0.0f));
	matrix cached = matrix(kernel_rows, std::vector<float>(kernel_cols, 0.0f));

	for (int i = 0; i < output->height; ++i) {
		int i_step = i * stride;
		for (int j = 0; j < output->width; ++j) {
			int j_step = j * stride;
			for (int ki = 0; ki < kernel_rows && i + ki < input->height; ki++) {
				for (int kj = 0; kj < kernel_cols && j + kj < input->width; kj++) {
					if ((i_step + ki) < padding_y || (j_step + kj) < padding_x || 
						(i_step + ki) > input->height || (j_step + kj) > input->width) {
							cached[ki][kj] = 0.5f;
					} else {
						cached[ki][kj] = input->values[i_step + ki - padding_y][j_step + kj - padding_x];
					}
				}
			}
			output->values[i][j] = activation(k->getActivationType(), k->getProductSum(cached));
		}
		cached = matrix(kernel_rows, std::vector<float>(kernel_cols, 0.0f));
	}
}

void dilationConvolution(CNNFeature* output, CNNFeature* input, Kernel* k, int stride) {
	int kernel_rows = k->getRows();
	int kernel_cols = k->getColumns();
	output->height = (input->height - kernel_rows) / stride + 1;
	output->width = (input->width - kernel_cols) / stride + 1;
	output->values = matrix(output->height, std::vector<float>(output->width, 0.0f));
	matrix cached = matrix(kernel_rows, std::vector<float>(kernel_cols, 0.0f));

	for (int i = 0; i < output->height; ++i) {
		int i_step = i * stride;
		for (int j = 0; j < output->width; ++j) {
			int j_step = j * stride;
			for (int ki = 0; ki < kernel_rows && (i_step + ki) < input->height; ki += 2) {
				for (int kj = 0; kj < kernel_cols && (j_step + kj) < input->width; kj += 2) {
					cached[ki][kj] = input->values[i_step + ki][j_step + kj];
				}
			}
			output->values[i][j] = activation(k->getActivationType(), k->getProductSum(cached));
		}
		cached = matrix(kernel_rows, std::vector<float>(kernel_cols, 0.0f));
	}
}

void paddedDilationConvolution(CNNFeature* output, CNNFeature* input, Kernel* k, int stride) {
	int kernel_rows = k->getRows();
	int kernel_cols = k->getColumns();
	const int padding_y = kernel_rows / 2;
	const int padding_x = kernel_cols / 2;
	output->height = input->height / stride + 1;
	output->width = input->width / stride + 1;
	output->values = matrix(output->height, std::vector<float>(output->width, 0.0f));
	matrix cached = matrix(kernel_rows, std::vector<float>(kernel_cols, 0.0f));

	for (int i = 0; i < output->height; ++i) {
		int i_step = i * stride;
		for (int j = 0; j < output->width; ++j) {
			int j_step = j * stride;
			for (int ki = 0; ki < kernel_rows && i + ki < input->height; ki += 2) {
				for (int kj = 0; kj < kernel_cols && j + kj < input->width; kj += 2) {
					if ((i_step + ki) < padding_y || (j_step + kj) < padding_x ||
						(i_step + ki) > input->height || (j_step + kj) > input->width) {
						cached[ki][kj] = 0.5f;
					}
					else {
						cached[ki][kj] = input->values[i_step + ki - padding_y][j_step + kj - padding_x];
					}
				}
			}
			output->values[i][j] = activation(k->getActivationType(), k->getProductSum(cached));
		}
		cached = matrix(kernel_rows, std::vector<float>(kernel_cols, 0.0f));
	}
}
