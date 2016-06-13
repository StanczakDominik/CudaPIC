#ifndef HELPERS_H_
#define HELPERS_H_

#include <iostream>
#include <stdio.h>

#define ELECTRON_MASS 1.0f
#define PROTON_MASS 1860.0f
#define ELECTRON_CHARGE 1.0f
#define EPSILON_ZERO 1.0f
#define L 1

void CUDA_ERROR( cudaError_t err);

#define position_to_grid_index(x, dx) ((int)(x/dx))
#define position_in_cell(x, dx) ((float)(x-dx*(int)(x/dx)))
#define ijk_to_n(i, j, k, N_grid) (N_grid * N_grid * (k%N_grid) + N_grid * (j%N_grid) + (i%N_grid))
#endif
