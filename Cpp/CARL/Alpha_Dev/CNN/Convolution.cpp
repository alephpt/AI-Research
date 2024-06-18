#include "Convolution.h"

Convolution::Convolution() : stride(1), n_kernels(0), kernels(vector<Kernel*>(0)) {}
Convolution::~Convolution() { kernels.clear(); }


int Convolution::getStride() { return stride; }
int Convolution::getKernelCount() { return n_kernels; }
vector<Kernel*> Convolution::getKernels() { return kernels; }
void Convolution::setStride(int s) { stride = s; }
void Convolution::addNewKernel(Kernel* k) { 
	kernels.push_back(k); 
	n_kernels++;
}

ftensor3d Convolution::convolutionFunction(ftensor3d f, int n_samples)
{
	int yod = (int)f[0].size();
	int xi = (int)f[0][0].size();
	ftensor3d output;

	// for each kernel
	for (int k = 0; k < n_kernels; k++) {
		int k_w = kernels[k]->getFilterWidth();
		int k_h = kernels[k]->getFilterHeight();
		int out_h = (yod - k_h + 1) / stride;
		int out_w = (xi - k_w + 1) / stride;
		fmatrix weights = kernels[k]->getFilter()->weights;
		fmatrix data = fmatrix(out_h, fvector(out_w, -1.0f));

		// for each output row + stride
		for (int y = 0; y < out_h; ++y) {
			int y_step = y * stride;
			// for each output column + stride
			for (int x = 0; x < out_w; ++x) {
				int x_step = x * stride;
				float product_sum = 0.0f;
				int product_count = 0;
				// for each kernel row + step < input height
				for (int k_y = 0; k_y < k_h && (k_y + y_step) < yod; k_y++) {
					// for each kernel column + step < input width
					for (int k_x = 0; k_x < k_w && (k_x + x_step) < xi; k_x++) {										
						// for each sample
						for (int z = 0; z < n_samples; z++) {
							// TODO: Kernel COULD be a Tensor
							product_sum += f[z][(size_t)k_y + y_step][(size_t)k_x + x_step] * weights[k_y][k_x];
							product_count++;
						}
					}
				}
				data[y / stride][x / stride] = product_sum / product_count;
			}
		}
		output.push_back(data);
	}
	
	return output;
}
