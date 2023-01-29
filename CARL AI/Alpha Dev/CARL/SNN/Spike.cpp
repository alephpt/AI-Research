#include "Spike.h"
#include "../Types/General.h"
#include <stdio.h>


void initSpike(Spike* s, float a) {
    s->fired = false;
    s->amplitude = a;
    s->timestamp = getTime();
}

void printSpike(Spike* s, int n) {
    printf("\t- Spike %d:\n", n);
    printf("\t\tfired: \t\t%s\n", s->fired ? "true" : "false");
    printf("\t\ttime: \t\t%.9lf\n", s->timestamp);
    printf("\t\tamplitude: \t%f\n", s->amplitude);
}

Spike* createNewSpike(Spike* s, float a) {
    initSpike(s, a);
    return s;
}