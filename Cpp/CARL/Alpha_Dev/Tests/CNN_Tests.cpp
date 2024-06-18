#include "CNN_Tests.h"
#include <vector>
#include <stdio.h>
#include "../Tests/TestData.h"
#include "../Data/Load_Data.h"

void testKernelParameters() {
    printf("Kernel Init:\n");
    Kernel k = Kernel();
    k.printFilter();

    k.setFilterStyle(RIGHT_EDGE_FILTER);
    k.printFilter();

    k.setFilterParameters(NxTHREE, 4, TOP_EDGE_FILTER);
    k.printFilter();

    k.setFilterStyle(LEFT_EDGE_FILTER);
    k.printFilter();

    k.setFilterStyle(BOTTOM_EDGE_FILTER);
    k.printFilter();

    k.setFilterParameters(FIVExFIVE, TOP_RIGHT_CORNER_FILTER);
    k.printFilter();

    k.setFilterStyle(BOTTOM_RIGHT_CORNER_FILTER);
    k.printFilter();

    k.setFilterStyle(BOTTOM_LEFT_CORNER_FILTER);
    k.printFilter();

    k.setFilterStyle(TOP_LEFT_CORNER_FILTER);
    k.printFilter();

    k.setFilterParameters(SEVENxSEVEN, ASCENDING_FILTER);
    k.printFilter();

    k.setFilterStyle(DESCENDING_FILTER);
    k.printFilter();
    
    k.setFilterStyle(VERTICAL_ASCENDING_FILTER);
    k.printFilter();

    k.setFilterStyle(VERTICAL_DESCENDING_FILTER);
    k.printFilter();
    
    k.setFilterStyle(LtoR_GRADIENT_FILTER);
    k.printFilter();
    
    k.setFilterStyle(RtoL_GRADIENT_FILTER);
    k.printFilter();
    
    k.setFilterStyle(TtoB_GRADIENT_FILTER);
    k.printFilter();
    
    k.setFilterStyle(BtoT_GRADIENT_FILTER);
    k.printFilter();

    k.setFilterParameters(ELEVENxELEVEN, TLtoBR_GRADIENT_FILTER);
    k.printFilter();
    
    k.setFilterStyle(BLtoTR_GRADIENT_FILTER);
    k.printFilter();

    k.setFilterStyle(GAUSSIAN_FILTER);
    k.printFilter();
    
    k.setFilterStyle(BALANCED_GAUSSIAN_FILTER);
    k.printFilter();
    
    k.setFilterStyle(NEGATIVE_GAUSSIAN_FILTER);
    k.printFilter();
    
    k.setFilterStyle(MODIFIED_GAUSSIAN_FILTER);
    k.printFilter();
 
    k.setFilterStyle(CONICAL_FILTER);
    k.printFilter();
     
    k.setFilterStyle(INVERSE_CONICAL_FILTER);
    k.printFilter();
}

void testCNNinit() {
    printf("Initializing CNN...\n\n");
    
    CNN* cnn = new CNN();
    
    cnn->addNewLayer(CNN_CONVOLUTION_LAYER);
    cnn->addNewKernel(THREExN, 1, TOP_EDGE_FILTER);
    cnn->addNewKernel(NxTHREE, 1, RIGHT_EDGE_FILTER);
    cnn->addNewKernel(THREExTHREE, TLtoBR_GRADIENT_FILTER);
    cnn->addNewKernel(THREExTHREE, BLtoTR_GRADIENT_FILTER);
    
    cnn->addNewLayer(CNN_POOLING_LAYER);                            // TODO: Make Sure We Don't Add Kernels to a Pooling Layer

    cnn->addNewLayer(CNN_CONVOLUTION_LAYER);
    cnn->addNewKernel(THREExTHREE, GAUSSIAN_FILTER);
    cnn->addNewKernel(THREExTHREE, CONICAL_FILTER);
    cnn->addNewKernel(THREExN, 1, LtoR_GRADIENT_FILTER);
    cnn->addNewKernel(NxTHREE, 1, TtoB_GRADIENT_FILTER);

    cnn->addNewLayer(CNN_POOLING_LAYER);
    cnn->setPoolType(L2_POOLING);

    cnn->addNewLayer(CNN_FLATTENING_LAYER);

    cnn->addNewLayer(CNN_FULLY_CONNECTED_LAYER);

    cnn->printCNN();
}

void testConvolutions() {
    printf("Convolution Test Started.\n");
    FixedFilterDimensions fd1 = THREExTHREE;
    FixedFilterDimensions fd2 = THREExTHREE;
    FixedFilterDimensions fd3 = THREExTHREE;
    FixedFilterDimensions fd4 = THREExTHREE;
    FixedFilterDimensions fd5 = THREExTHREE;
    FixedFilterDimensions fd6 = THREExTHREE;
    FilterStyle fs1 = TOP_EDGE_FILTER;
    FilterStyle fs2 = RIGHT_EDGE_FILTER;
    FilterStyle fs3 = TLtoBR_GRADIENT_FILTER;
    FilterStyle fs4 = BLtoTR_GRADIENT_FILTER;
    FilterStyle fs5 = CONICAL_FILTER;
    FilterStyle fs6 = INVERSE_CONICAL_FILTER;

    printf("Initializing CNN, Convolution Layer and Input Data:\n");
    CNN* cnn = new CNN();
    cnn->addNewInputSample(ftensor3d(4, smiles));

    cnn->addNewLayer(CNN_CONVOLUTION_LAYER);

    cnn->addNewKernel(fd1, fs1);
    cnn->addNewKernel(fd2, fs2);
    cnn->addNewKernel(fd3, fs3);
    cnn->addNewKernel(fd4, fs4);
    cnn->addNewKernel(fd5, fs5);
    cnn->addNewKernel(fd6, fs6);

    cnn->printCNN();

    printf("\nPooling Test Started.\n");

    printf("Input:\n");
    printf("pooling Height: %d\n", (int)smiles.size());
    printf("pooling Width: %d\n", (int)smiles[0].size());
    printFMatrix(smiles);

    ftensor3d conv_out = cnn->Convolute();

    printf("\n%s %s:\n", filterDimensionsString.at(fd1).c_str(), filterStyleString[fs1].c_str());
    printFMatrix(conv_out[0]);

    printf("\n%s %s:\n", filterDimensionsString.at(fd2).c_str(), filterStyleString[fs2].c_str());
    printFMatrix(conv_out[1]);

    printf("\n%s %s:\n", filterDimensionsString.at(fd3).c_str(), filterStyleString[fs3].c_str());
    printFMatrix(conv_out[2]);

    printf("\n%s %s:\n", filterDimensionsString.at(fd4).c_str(), filterStyleString[fs4].c_str());
    printFMatrix(conv_out[3]);

    printf("\n%s %s:\n", filterDimensionsString.at(fd5).c_str(), filterStyleString[fs5].c_str());
    printFMatrix(conv_out[4]);

    printf("\n%s %s:\n", filterDimensionsString.at(fd6).c_str(), filterStyleString[fs6].c_str());
    printFMatrix(conv_out[5]);
}

void testPooling() {
    fmatrix pool_results;
    PoolingStyle pool_method = MAX_POOLING;

    printf("Initializing CNN, Pooling Layer and Input Data:\n");
    
    CNN* cnn = new CNN();
    cnn->addNewLayer(CNN_POOLING_LAYER);
    cnn->addNewInputSample(ftensor3d(1, smiles));
    cnn->printCNN();

    printf("\nPooling Test Started.\n");

    printf("Input:\n");
    printf("pooling Height: %d\n", (int)smiles.size());
    printf("pooling Width: %d\n", (int)smiles[0].size());
    printFMatrix(smiles);

    cnn->setPoolType(pool_method);
    pool_results = cnn->Pooling();

    printf("\n%s output:\n", poolingStyleString[pool_method].c_str());
    printf("pooling Height: %d\n", (int)pool_results.size());
    printf("pooling Width: %d\n", (int)pool_results[0].size());
    printFMatrix(pool_results);
    pool_results.clear();

    pool_method = AVG_POOLING;
    cnn->setPoolType(pool_method);
    pool_results = cnn->Pooling();

    printf("\n%s output:\n", poolingStyleString[pool_method].c_str());
    printf("pooling Height: %d\n", (int)pool_results.size());
    printf("pooling Width: %d\n", (int)pool_results[0].size());
    printFMatrix(pool_results);
    pool_results.clear();

    pool_method = L2_POOLING;
    cnn->setPoolType(pool_method);
    pool_results = cnn->Pooling();

    printf("\n%s output:\n", poolingStyleString[pool_method].c_str());
    printf("pooling Height: %d\n", (int)pool_results.size());
    printf("pooling Width: %d\n", (int)pool_results[0].size());
    printFMatrix(pool_results);
    pool_results.clear();

    pool_method = STOCHASTIC_POOLING;
    cnn->setPoolType(pool_method);
    pool_results = cnn->Pooling();

    printf("\n%s output:\n", poolingStyleString[pool_method].c_str());
    printf("pooling Height: %d\n", (int)pool_results.size());
    printf("pooling Width: %d\n", (int)pool_results[0].size());
    printFMatrix(pool_results);
    pool_results.clear();

    pool_method = GLOBAL_MAX_POOLING;
    cnn->setPoolType(pool_method);
    pool_results = cnn->Pooling();

    printf("\n%s output:\n", poolingStyleString[pool_method].c_str());
    printf("pooling Height: %d\n", (int)pool_results.size());
    printf("pooling Width: %d\n", (int)pool_results[0].size());
    printFMatrix(pool_results);
    pool_results.clear();

    pool_method = GLOBAL_AVG_POOLING;
    cnn->setPoolType(pool_method);
    pool_results = cnn->Pooling();

    printf("\n%s output:\n", poolingStyleString[pool_method].c_str());
    printf("pooling Height: %d\n", (int)pool_results.size());
    printf("pooling Width: %d\n", (int)pool_results[0].size());
    printFMatrix(pool_results);
    pool_results.clear();
}