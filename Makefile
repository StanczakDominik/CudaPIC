NVCC = nvcc
NVCC_FLAGS = -arch=sm_20 -rdc=true  --use_fast_math -lcufft -g -G

all: main.out

main.out: main.o particles.o grid.o helpers.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

main.o: main.cu grid.o particles.o helpers.o
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

particles.o: particles.cu particles.cuh grid.cuh helpers.o
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

grid.o: grid.cu grid.cuh helpers.o
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

helpers.o: helpers.cu helpers.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe *.out data/*.dat data/*.bdat

rerun:
	rm  data/*.png data/*.dat

run:
	make
	./main.out &
	sleep 3
	python particles_plot.py
