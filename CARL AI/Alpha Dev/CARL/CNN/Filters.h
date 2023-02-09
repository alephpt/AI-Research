#pragma once

void oneXone(int* r, int* c) { *r = 1; *c = 1; }
void twoXtwo(int* r, int* c) { *r = 2; *c = 2; }
void threeXthree(int* r, int* c) { *r = 3; *c = 3; }
void fiveXfive(int* r, int* c) { *r = 5; *c = 5; }
void sevenXseven(int* r, int* c) { *r = 7; *c = 7; }
void elevenXeleven(int* r, int* c) { *r = 11; *c = 11; }
void oneXn(int* r, int* c, int n) { *r = 1; *c = n; }
void twoXn(int* r, int* c, int n) { *r = 2; *c = n; }
void threeXn(int* r, int* c, int n) { *r = 3; *c = n; }
void nXone(int* r, int* c, int n) { *r = n; *c = 1; }
void nXtwo(int* r, int* c, int n) { *r = n; *c = 2; }
void nXthree(int* r, int* c, int n) { *r = n; *c = 3; }