NVCC = nvcc
NVCC_FLAGS = -arch=sm_20  --use_fast_math -lcufft

all: main.out

main.out: main.o particles.o grid.o helpers.o
	$(NVCC) $^ -o $@

main.o: main.cu grid.cuh particles.cuh helpers.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

particles.o: particles.cu particles.cuh grid.cuh helpers.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

grid.o: grid.cu grid.cuh helpers.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

helpers.o: helpers.cu helpers.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
