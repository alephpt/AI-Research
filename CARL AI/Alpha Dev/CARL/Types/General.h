#pragma once
#include <vector>

double getTime();
std::vector<std::vector<float>> generate2dNoise(int rows, int cols);
void print2DVector(std::vector<std::vector<float>> values, int rows, int cols);