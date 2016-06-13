#ifndef GRID_H_
#define GRID_H_

#include "helpers.cuh"
#include <cufft.h>

struct Grid{
    float *rho;
    float *Ex;
    float *Ey;
    float *Ez;

    float *d_rho;
    float *d_Ex;
    float *d_Ey;
    float *d_Ez;

    //fourier transformed versions of grid quantities, for fields solver
    cufftComplex *d_F_rho;
    cufftComplex *d_F_Ex;
    cufftComplex *d_F_Ey;
    cufftComplex *d_F_Ez;

    //instructions for cuFFT
    cufftHandle plan_forward;
    cufftHandle plan_backward;

    //the wave vector, for the field solver
    float *kv;
    float *d_kv;
};

__global__ void solve_poisson(float *d_kv, cufftComplex *d_F_rho, cufftComplex *d_F_Ex, cufftComplex *d_F_Ey, cufftComplex *d_F_Ez, int N_grid, int N_grid_all);
__global__ void real2complex(float *input, cufftComplex *output, int N_grid, int N_grid_all);
__global__ void complex2real(cufftComplex *input, float *output, int N_grid, int N_grid_all);
__global__ void scale_down_after_fft(float *d_Ex, float *d_Ey, float *d_Ez, int N_grid, int N_grid_all);
__global__ void set_grid_array_to_value(float *arr, float value, int N_grid, int N_grid_all);

void init_grid(Grid *g, int N_grid, int N_grid_all);
void field_solver(Grid *g, int N_grid, int N_grid_all, dim3 gridBlocks, dim3 gridThreads);
void dump_density_data(Grid *g, char* name, int N_grid, int N_grid_all);
void dump_running_density_data(Grid *g, char* name, int N_grid, int N_grid_all);

void debug_field_solver_uniform(Grid *g, int N_grid, int N_grid_all);
void debug_field_solver_sine(Grid *g, int N_grid, int N_grid_all);

#endif
