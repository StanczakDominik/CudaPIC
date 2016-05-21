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
"""


the_line = "#define N_particles_1_axis {}\n"

with open(code_file.format("")) as f:
    code = f.read()
    # print(code)

for N_threads in np.arange(24, 71):
    code_with_benchmark = code_start + the_line.format(N_threads) + code
    with open(code_file.format(N_threads), "w") as f:
        f.write(code_with_benchmark)
    compile_line = compile_command.format(N_threads, N_threads, N_threads)
    print(compile_line)
    os.system(compile_line)
