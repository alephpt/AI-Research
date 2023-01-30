#include "Membrane.h"
#include "../Types/General.h"
#include <stdio.h>


void initMembrane(Membrane* m, float capacitance, float resistance, float V_rest, float V_threshold) {
    m->capacitance = capacitance;
    m->resistance = resistance;
    m->V_rest = V_rest;
    m->V_threshold = V_threshold;
    m->t = getTime();
}

Membrane* createNewMembrane(Membrane* m, float c, float r, float V_rest, float V_thresh) {
    initMembrane(m, c, r, V_rest, V_thresh);
    return m;
}

void printMembrane (Membrane* m) {
    printf("Membrane:\n");
    printf("\tcapacitance: \t\t%f\n", m->capacitance);
    printf("\tresistance: \t\t%f\n", m->resistance);
    printf("\tV_rest: \t\t%f\n", m->V_rest);
    printf("\tV_threshold: \t\t%f\n", m->V_threshold);
    printf("\tt: \t\t\t%.9lf\n", m->t);
}