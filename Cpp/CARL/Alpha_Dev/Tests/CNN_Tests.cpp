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
    CNN* cnn = new CNN();
    cnn->addNewLayer(CNN_CONVOLUTION_LAYER);
    printf("Convolution Test Started.\n");

    cnn->addNewInputSample(ftensor3d(1, smiles));
    cnn->printCNN();

    // cnn->Pool();
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