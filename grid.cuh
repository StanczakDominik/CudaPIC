#ifndef GRID_H_
#define GRID_H_

#define N_grid 16
#define N_grid_all (N_grid *N_grid * N_grid)
#define dx (L/float(N_grid))
#define dy dx
#define dz dx
#define L 1e-4

#include "helpers.cuh"
#include <cufft.h>



extern dim3 gridThreads;
extern dim3 gridBlocks;

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
    cufftComplex *d_fourier_rho;
    cufftComplex *d_fourier_Ex;
    cufftComplex *d_fourier_Ey;
    cufftComplex *d_fourier_Ez;

    //instructions for cuFFT
    cufftHandle plan_forward;
    cufftHandle plan_backward;

    //the wave vector, for the field solver
    float *kv;
    float *d_kv;
};

__global__ void solve_poisson(float *d_kv, cufftComplex *d_fourier_rho, cufftComplex *d_fourier_Ex, cufftComplex *d_fourier_Ey, cufftComplex *d_fourier_Ez);
__global__ void real2complex(float *input, cufftComplex *output);
__global__ void complex2real(cufftComplex *input, float *output);
__global__ void scale_down_after_fft(float *d_Ex, float *d_Ey, float *d_Ez);
__global__ void set_grid_array_to_value(float *arr, float value);

void init_grid(Grid *g);
void debug_field_solver_uniform(Grid *g);
void debug_field_solver_sine(Grid *g);
void field_solver(Grid *g);
void dump_density_data(Grid *g, char* name);
void dump_running_density_data(Grid *g, char* name);


__device__ int position_to_grid_index(float X);
__device__ float position_in_cell(float x);

#endif
