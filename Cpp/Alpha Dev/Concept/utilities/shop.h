#ifndef SHOP_H
#define SHOP_H

#include <assert.h>

float** transpose(float** a, int n_rows, int n_cols);

float sigmoid(float x);
float sigmoid_derivative(float x);
void sigmoidLayer(float* layer, int size);

void add1DArray(float* array1, float* array2, int size);
void add1DArrayLayer(float* dotProductResult, float* bias, int size, float* layer);
float* allocate1DArray(int n_filters);
void deallocate1DArray(float* array);
void subtract1DArray(float* arr1, float* arr2, int size);

void normalize(float* array, int size);

float** allocate2DArray(int rows, int cols);
void deallocate2DArray(float** array, int rows);
float** subtract2DArray(float** array1, int rows1, int cols1, float** array2, int rows2, int cols2);

float** allocateMatrix(int batch_size, int n_inputs);
void freeMatrix(float** matrix, int n_rows);
float** generateRandomNoise(int n_inputs, int batch_size);
void serializeMatrix(float** matrix, int rows, int cols, char* buffer);
void deserializeMatrix(char* buffer, float** matrix);

void scalarMultiply(float scalar, float* array, int size);
void scalarMultiply2D(float scalar, float** array, int rows, int cols);
void elementWiseMultiply(float* a, float* b, int n);
float dotProduct(float* a, float* b, int length);
float** dot(float** matrix1, float** matrix2, int rows1, int cols1, int cols2);

#endif