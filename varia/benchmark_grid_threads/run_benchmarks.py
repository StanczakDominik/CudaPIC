import os
import numpy as np


code_file = "/home/dominik/Code/CUDA/CUDAPIC/benchmark_particles{}.cu"
compile_command = "nvcc -arch=sm_20  --use_fast_math -lcufft benchmark_particles{}.cu -o benchmark_particles{}.out; ./benchmark_particles{}.out"

code_start = """#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <iostream>
using namespace std;

#define ELECTRON_MASS 9.10938356e-31
#define PROTON_MASS 1.6726219e-27
#define ELECTRON_CHARGE 1
// NOTE: setting electron charge to the default SI 1.6e-19 value breaks interpolation
#define EPSILON_ZERO 8.854e-12

//TODO: THIS HERE TIMESTEP I AM NOT COMPLETELY CERTAIN ABOUT
#define NT 1000
#define N_grid 32
#define N_particles_1_axis 64
#define N_particles  (N_particles_1_axis*N_particles_1_axis*N_particles_1_axis)
#define L 1e-4
#define dt 1e-25
#define N_grid_all (N_grid *N_grid * N_grid)
#define dx (L/float(N_grid))
#define dy dx
#define dz dx


dim3 particleThreads(512);
dim3 particleBlocks((N_particles+particleThreads.x - 1)/particleThreads.x);
"""


the_line = "dim3 gridThreads({},{},{});\n"

with open(code_file.format("")) as f:
    code = f.read()
    # print(code)

for N_threads in np.arange(2, 11):
    code_with_benchmark = code_start + the_line.format(N_threads, N_threads, N_threads) + code
    with open(code_file.format(N_threads), "w") as f:
        f.write(code_with_benchmark)
    compile_line = compile_command.format(N_threads, N_threads, N_threads)
    print(compile_line)
    os.system(compile_line)
