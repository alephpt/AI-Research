#include "General.h"
#include <chrono>
#include <random>
#include <cmath>


double getTime() 
{
    auto now = std::chrono::high_resolution_clock::now();
    double nanoseconds = std::chrono::duration<double, std::nano>(now.time_since_epoch()).count();
    return (static_cast<double>(nanoseconds) / 1000000000.0);
}


std::vector<std::vector<float>> generate2dNoise(int rows, int cols)
{
    std::vector<std::vector<float>>vals(rows, std::vector<float>(cols, 0.0f));
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            vals[i][j] = dis(gen);
        }
    }

    return vals;
}

void print2DVector(std::vector<std::vector<float>> values, int rows, int cols) 
{
    for (int i = 0; i < rows; i++)
    {
        printf("\t\t");
        for (int j = 0; j < cols; j++)
        {
            printf("[%.2f] ", values[i][j]);
        }
        printf("\n");
    }
}