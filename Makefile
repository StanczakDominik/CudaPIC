NVCC = nvcc
NVCC_FLAGS = -arch=sm_20 --ptxas-options=-v --use_fast_math -lcufft -g -G
main.out: main.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@
