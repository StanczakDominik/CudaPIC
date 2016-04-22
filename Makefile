main: main.cu
	nvcc -o main main.cu -arch=sm_20
