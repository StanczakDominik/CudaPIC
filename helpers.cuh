#ifndef HELPERS_H_
#define HELPERS_H_

#include <iostream>
#include <stdio.h>

#define NT 10000
#define dt 0.01f

// #define ELECTRON_MASS 9.10938356e-31
// #define PROTON_MASS 1.6726219e-27
#define ELECTRON_MASS 1.0f
#define PROTON_MASS 1860.0f
#define ELECTRON_CHARGE 1.0f
// NOTE: setting electron charge to the default SI 1.6e-19 value breaks interpolation
// #define EPSILON_ZERO 8.854e-12
#define EPSILON_ZERO 1.0f

void CUDA_ERROR( cudaError_t err);

#endif
