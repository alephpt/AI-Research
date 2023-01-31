#include "CNN_Tests.h"
#include <vector>
#include <stdio.h>

static inline void printKernel(Kernel* k) {
    for (int y = 0; y < k->rows; y++) {
        printf("\t\t");
        for (int x = 0; x < k->columns; x++) {
            printf("[%.2f] ", k->values[y][x]);
        }
        printf("\n");
    }
}

static void testKernelFilter(FilterDimensions filter) {
    Kernel* k = initKernel(filter);

    printf("%s Kernel has %d k->rows and %d k->cols\n", filterString[filter].c_str(), k->rows, k->columns);
    printKernel(k);
    printf("\n");

    k->values.clear();
    delete k;
}

static void testKernelFilter(FilterDimensions filter, int n) {
    Kernel* k = initKernel(filter, n);

    printf("%s Kernel has %d k->rows and %d k->cols\n", filterString[filter].c_str(), k->rows, k->columns);
    printKernel(k);
    printf("\n");

    k->values.clear();
    delete k;
}

void testKernelInit() {
    testKernelFilter(FIVExFIVE);
    testKernelFilter(THREExTHREE);
    testKernelFilter(THREExONE);
    testKernelFilter(ONExTHREE);
    testKernelFilter(ONExN, 7);
    testKernelFilter(NxONE, 9);
    
    return;
}