#include "../Types/General.h"
#include "Spike.h"
#include <stdio.h>


void initSpike(Spike* s, fscalar a) {
    s->fired = false;
    s->amplitude = a;
    s->timestamp = getTime();
}

void printSpike(Spike* s, int n) {
    printf("\t- Spike %d:\t", n);
    printf("\tfired: \t\t%s", s->fired ? "true" : "false");
    printf("\t\tamplitude: \t%f", s->amplitude);
    printf("\ttime: \t%.9lf\n", s->timestamp);
}

Spike* createNewSpike(Spike* s, fscalar a) {
    initSpike(s, a);
    return s;
}