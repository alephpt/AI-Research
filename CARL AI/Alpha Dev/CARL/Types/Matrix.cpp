#include "Matrix.h"
#include "General.h"

void printFMatrix(fmatrix values) {
    size_t rows = values.size();
    size_t cols = values[0].size();

    for (int i = 0; i < rows; i++)
    {
        printf("\t\t");
        for (int j = 0; j < cols; j++)
        {
            printColor(values[i][j]);
        }
        printf("\n");
    }
}