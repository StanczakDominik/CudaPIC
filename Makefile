main: main.cu
	nvcc -o main.out main.cu -arch=sm_20
