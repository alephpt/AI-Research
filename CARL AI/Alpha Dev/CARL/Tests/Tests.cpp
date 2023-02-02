#include "Tests.h"

void runTests() {
        // Primitive Tests //
    // testRandomNumbers();
    // testVector();
    // testMatrix();
    // test3DTensor();
    // test4DTensor();
    // testActivationType();

        // CNN Tests //
    // testKernelInit();              // TODO: Fix 1x1, Nx1 and 1xN
    // testAdjustKernelDims();
    // testConvolutionInit();
    // testConvolutionFilters();
    // testConvolutionInput();
     testConvolutions();

        // SNN Tests //
    // testinitSNN();
    // testConnectivityMatrix();
    // testGeneratorinit();

    return;
}
