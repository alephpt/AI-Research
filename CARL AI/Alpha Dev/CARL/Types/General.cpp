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

void printColor(float in) {
    if (in < -0.985f) {
        // magenta
        printf("\033[0;35m[%07.3lf] \033[0m", in);
    }
    else
    if (in < -0.67f) {
        // red
        printf("\033[0;31m[%07.3lf] \033[0m", in);
    }
    else
    if (in < -0.35f) {
        // bright red
        printf("\033[0;91m[%07.3lf] \033[0m", in);
    }
    else
    if (in < -0.05f) {
        // dark yellow
        printf("\033[0;33m[%07.3lf] \033[0m", in);
    }
    else
    if (in < 0.05) {
        // dark green
        printf("\033[0;32m[%07.3lf] \033[0m", in);
    }
    else
    if (in < 0.29f) {
        // dark blue
        printf("\033[0;34m[%07.3lf] \033[0m", in);
    }
    else
    if (in < 0.6f) {
        // bright blue
        printf("\033[0;94m[%07.3lf] \033[0m", in);
    }
    else
    if (in < 0.95f) {
        // bright cyan
        printf("\033[0;96m[%07.3lf] \033[0m", in);
    }
    else {
        printf("[%07.3lf] ", in);
    }
}

void print2DVector(std::vector<std::vector<float>> values, int rows, int cols) 
{
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