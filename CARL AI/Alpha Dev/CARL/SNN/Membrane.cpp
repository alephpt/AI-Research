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

extern Membrane* createNewMembrane(Membrane* m, float c, float r, float V_rest, float V_thresh) {
    initMembrane(m, c, r, V_rest, V_thresh);
    return m;
}

void printMembrane (Membrane* m) {
    printf("Membrane:");
    printf("\t\t\tcapacitance: \t%f\n", m->capacitance);
    printf("\t\t\t\tresistance: \t%f\n", m->resistance);
    printf("\t\t\t\tV_rest: \t%f\n", m->V_rest);
    printf("\t\t\t\tV_threshold: \t%f\n", m->V_threshold);
    printf("\t\t\t\tt: \t\t%.9lf\n", m->t);
}