#pragma once
#include "Kernel.h"

// TODO: Look at refactoring Kernel into Convolution
class Convolution {
public:
    Convolution();
    ~Convolution();

    int getStride();
    int getKernelCount();
    vector<Kernel*> getKernels();

    void setStride(int);
    void addNewKernel(Kernel*);

    
    ftensor3d convolutionFunction(ftensor3d, int);
private:
    int stride;
    int n_kernels;
    vector<Kernel*> kernels;
};