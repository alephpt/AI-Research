#include "Neuron.h"
#include <math.h>

float membranePotential(float inputCurrent, Membrane membrane) {
    return membrane.V_rest + (membrane.V_threshold - membrane.V_rest) * (1.0f - expf(-membrane.t / (membrane.resistance * membrane.capacitance)));
}