import matplotlib.pyplot as plt
import numpy as np
import os
benchmark_directory = "/home/dominik/Code/CUDA/CUDAPIC/benchmark_grid/"
list_files = os.listdir(benchmark_directory)

list_N = []
list_T = []
list_B = []
list_RT = []

for filename in list_files:
    if (filename[-4:] == "bdat"):
        filename = benchmark_directory + filename
        with open(filename) as f:
            f.readline()
            data = np.loadtxt(f)
            print(data)
            N_particles, Threads, Blocks, Runtime = data
            list_N.append(N_particles)
            list_T.append(Threads)
            list_B.append(Blocks)
            list_RT.append(Runtime/1000)

fig, (axN, axT, axB) = plt.subplots(3)
axN.set_title("Grid benchmark: CUDA Particle in Cell simulation: varying number of grid cells")
axN.plot((np.array(list_N))**(1/3), list_RT, "ko")
axN.set_xlabel("Number of grid cells per axis")
axN.set_ylabel("Runtime [s]")
axN.grid()


axT.plot(list_T, list_RT, "ko")
axT.set_xlabel("Number of threads per block")
axT.set_ylabel("Runtime [s]")
axT.grid()


axB.plot(list_B, list_RT, "ko")
axB.set_xlabel("Number of blocks in simulation")
axB.set_ylabel("Runtime [s]")
axB.grid()

plt.tight_layout()
plt.savefig("grid_benchmark_data.png")
plt.show()
