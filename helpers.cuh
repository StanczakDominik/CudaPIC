#ifndef HELPERS_H_
#define HELPERS_H_

#include <iostream>
#include <stdio.h>

#define NT 10000
#define dt 1e-6

#define ELECTRON_MASS 9.10938356e-31
#define PROTON_MASS 1.6726219e-27
#define ELECTRON_CHARGE 1
// NOTE: setting electron charge to the default SI 1.6e-19 value breaks interpolation
#define EPSILON_ZERO 8.854e-12


void CUDA_ERROR( cudaError_t err);

#endif
