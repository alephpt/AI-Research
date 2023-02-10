#include "GAN_Tests.h"
#include <stdio.h>

int n_in = 50;
int n_hid = 100;
int n_out = 50;


void testGeneratorinit() {
	Generator* g = new Generator(n_in, n_hid, n_out);
	
	printf("testGeneratorinit(%d, %d, %d)\n", n_in, n_hid, n_out);
	printf("g->n_inputs: \t%d\n", g->n_inputs);
	printf("g->n_hidden: \t%d\n", g->n_hidden);
	printf("g->n_bias: \t%d\n", g->n_outputs);
}