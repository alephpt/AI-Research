#include "General.h"
#include <chrono>


double getTime() {
    auto now = std::chrono::high_resolution_clock::now();
    double nanoseconds = std::chrono::duration<double, std::nano>(now.time_since_epoch()).count();
    return (static_cast<double>(nanoseconds) / 1000000000.0);
}