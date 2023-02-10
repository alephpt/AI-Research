#include <stdbool.h>
#include "shop.h"

    // ACTIVATION FUNCTIONS //

    // RELU //
float relu(float x) {
    if (x < 0) { return 0; }
    return x;
}

float relu_derivative(float x) {
    if (x < 0) {
        return 0;
    }
    return 1;
}


    // LEAKY_RELU //

float leaky_relu(float x, float alpha) {
    if (x > 0) {
        return x;
    }
    else {
        return alpha * x;
    }
}

float leaky_relu_derivative(float x, float alpha) {
    if (x > 0) {
        return 1;
    } else {
        return alpha;
    }
}

    // TANH //
float tanh(float x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

float tanh_derivative(float x) {
    return 1.0 - x*x;
}

    // SIGMOIDS //
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

void sigmoidLayer(float* layer, int size) {
    for (int i = 0; i < size; i++) {
        layer[i] = 1 / (1 + exp(-layer[i]));
    }
}

    


    // TRANSPOSITION //
float** transpose(float** a, int n_rows, int n_cols) {
    float** transposed = allocate2DArray(n_cols, n_rows);
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            transposed[j][i] = a[i][j];
        }
    }
    return transposed;
}


    // 1D ARRAYS //
float* allocate1DArray(int n_filters) {
    float* array = (float*)malloc(n_filters * sizeof(float));
    return array;
}

    void deallocate1DArray(float* array) {
        free(array);
    }

void add1DArray(float* array1, float* array2, int size) {
    for (int i = 0; i < size; i++) {
        array1[i] += array2[i];
    }
}

void add1DArrayLayer(float* dotProductResult, float* bias, int size, float* layer) {
    for (int i = 0; i < size; i++) {
        layer[i] = dotProductResult[i] + bias[i];
    }
}

void subtract1DArray(float* arr1, float* arr2, int size) {
    for (int i = 0; i < size; i++) {
        arr1[i] -= arr2[i];
    }
}

void normalize(float* array, int size) {
    int i;
    float sum = 0;
    for (i = 0; i < size; i++) {
        sum += array[i];
    }
    for (i = 0; i < size; i++) {
        array[i] /= sum;
    }
}


    // 2D ARRAYS //
float** allocate2DArray(int rows, int cols) {
    float** arr = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        arr[i] = (float*)malloc(cols * sizeof(float));
    }
    return arr;
}

void deallocate2DArray(float** array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

float** subtract2DArray(float** array1, int rows1, int cols1, float** array2, int rows2, int cols2) {
    if (rows1 != rows2 || cols1 != cols2) {
        printf("Error: cannot subtract arrays of different dimensions\n");
        return NULL;
    }

    // Allocate memory for the result array
    float** result = (float**)malloc(rows1 * sizeof(float*));
    for (int i = 0; i < rows1; i++) {
        result[i] = (float*)malloc(cols1 * sizeof(float));
    }

    // Subtract the values of array2 from array1 and store them in result
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols1; j++) {
            result[i][j] = array1[i][j] - array2[i][j];
        }
    }

    return result;
}

void conv2d(float* input, float*** filters, float** output, int n_inputs, int n_filters, int filter_size) {
    for (int i = 0; i < n_filters; i++) {
        for (int j = 0; j < n_inputs; j++) {
            for (int k = 0; k < filter_size; k++) {
                for (int l = 0; l < filter_size; l++) {
                    output[i][j] += input[j] * filters[i][j][k][l];
                }
            }
        }
    }
}


void maxpool2d(float** input, float** output, int n_filters, int pool_size) {
    for (int i = 0; i < n_filters; i++) {
        for (int j = 0; j < n_inputs / pool_size; j++) {
            for (int k = 0; k < n_inputs / pool_size; k++) {
                float max = input[i][j * pool_size + k];
                for (int l = 0; l < pool_size; l++) {
                    for (int m = 0; m < pool_size; m++) {
                        if (input[i][(j * pool_size + l) * n_inputs + k * pool_size + m] > max) {
                            max = input[i][(j * pool_size + l) * n_inputs + k * pool_size + m];
                        }
                    }
                }
                output[i][j * n_inputs / pool_size + k] = max;
            }
        }
    }
}


    // MATRICIES //
float** allocateMatrix(int batch_size, int n_inputs) {
    float** matrix = (float**)malloc(batch_size * sizeof(float*));
    for (int i = 0; i < batch_size; i++) {
        matrix[i] = (float*)malloc(n_inputs * sizeof(float));
    }
    return matrix;
}

void freeMatrix(float** matrix, int n_rows) {
    for (int i = 0; i < n_rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

float** generateRandomNoise(int n_inputs, int batch_size) {
    float** random_noise = allocateMatrix(batch_size, n_inputs);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_inputs; j++) {
            random_noise[i][j] = (float)rand() / RAND_MAX;
        }
    }
    return random_noise;
}

void convMatrix(float** input, float** filters, float** output, int n_inputs, int n_filters, int filter_size) {
    for (int filter = 0; filter < n_filters; filter++) {
        for (int i = 0; i < n_inputs - filter_size + 1; i++) {
            for (int j = 0; j < n_inputs - filter_size + 1; j++) {
                float sum = 0;
                for (int k = 0; k < filter_size; k++) {
                    for (int l = 0; l < filter_size; l++) {
                        sum += input[i + k][j + l] * filters[filter][k * filter_size + l];
                    }
                }
                output[filter][i * (n_inputs - filter_size + 1) + j] = sum;
            }
        }
    }
}

void maxpoolMatrix(float** input, float** output, int n_inputs, int pool_size) {
    for (int i = 0; i < n_inputs / pool_size; i++) {
        for (int j = 0; j < n_inputs / pool_size; j++) {
            float max_val = input[i * pool_size][j * pool_size];
            for (int k = 0; k < pool_size; k++) {
                for (int l = 0; l < pool_size; l++) {
                    max_val = fmax(max_val, input[i * pool_size + k][j * pool_size + l]);
                }
            }
            output[i][j] = max_val;
        }
    }
}

void serializeMatrix(float** matrix, int rows, int cols, char* buffer) {
    // Declare a variable to keep track of the current position in the buffer
    int currentPos = 0;

    // Write the number of rows and columns to the buffer
    memcpy(buffer + currentPos, &rows, sizeof(int));
    currentPos += sizeof(int);
    memcpy(buffer + currentPos, &cols, sizeof(int));
    currentPos += sizeof(int);

    // Write the matrix data to the buffer
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            memcpy(buffer + currentPos, &matrix[i][j], sizeof(float));
            currentPos += sizeof(float);
        }
    }
}


void deserializeMatrix(char* buffer, float** matrix) {
    // Declare variables to store the number of rows and columns
    int rows, cols;

    // Read the number of rows and columns from the buffer
    memcpy(&rows, buffer, sizeof(int));
    memcpy(&cols, buffer + sizeof(int), sizeof(int));

    // Allocate memory for the matrix
    matrix = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(cols * sizeof(float));
    }

    // Read the matrix data from the buffer
    int currentPos = 2 * sizeof(int);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            memcpy(&matrix[i][j], buffer + currentPos, sizeof(float));
            currentPos += sizeof(float);
        }
    }
}

    // MATHS //
void elementWiseMultiply(float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = a[i] * b[i];
    }
}


float dotProduct(float* a, float* b, int length) {
    float result = 0;

    for (int i = 0; i < length; i++) {
        result += a[i] * b[i];
    }

    return result;
}

float** dot(float** matrix1, float** matrix2, int rows1, int cols1, int cols2) {
    float** result = allocate2DArray(rows1, cols2);
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            for (int k = 0; k < cols1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return result;
}

void scalarMultiply(float scalar, float* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = array[i] * scalar;
    }
}

void scalarMultiply2D(float scalar, float** array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i][j] = array[i][j] * scalar;
        }
    }
}