main: main.cu
	nvcc -o main.out main.cu -arch=sm_20 --ptxas-options=-v --use_fast_math -lcufft
