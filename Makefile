NVCC = nvcc
NVCC_FLAGS = -arch=sm_20  --use_fast_math -lcufft
main.out: main.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@
benchmark_particles.out: benchmark_particles.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@
