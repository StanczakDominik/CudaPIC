test: atomic_test.cu
	nvcc -o atomic_test.out atomic_test.cu -arch=sm_20
thrust_test: thrust_test.cu
	nvcc -o thrust_test.out thrust_test.cu -arch=sm_20
test_arrays: test_aos.cu
	nvcc -o test_aos.out test_aos.cu -arch=sm_20
test_soa: test_soa.cu
	nvcc -o test_soa.out test_soa.cu -arch=sm_20
test_aos: test_aos.cu
	nvcc -o test_aos.out test_aos.cu -arch=sm_20
test_memcpy: test_memcpy.cu
	nvcc -o test_memcpy.out test_memcpy.cu -arch=sm_20
2darray: 2darray.cu
	nvcc -o 2darray.out 2darray.cu -arch=sm_20
