#include <stdio.h>
/*CUDA error wraper*/
static void CUDA_ERROR( cudaError_t err)
{
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s, exiting\n", cudaGetErrorString(err));
        exit(-1);
    }
}


struct two_numbers
{
    int i;
    int j;
};

__global__ void increment(two_numbers *d_numbers, int N)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<N)
    {
        d_numbers[i].i *= 2;
        d_numbers[i].j *= -3;
    }
}

int main()
{
    int N = 10;
	/*get info on our GPU, defaulting to first one*/
	cudaDeviceProp prop;
	CUDA_ERROR(cudaGetDeviceProperties(&prop,0));
	printf("Found GPU '%s' with %g Gb of global memory, max %d threads per block, and %d multiprocessors\n",
		prop.name, prop.totalGlobalMem/(1024.0*1024.0),
		prop.maxThreadsPerBlock,prop.multiProcessorCount);

    two_numbers *numbers = new two_numbers[N];
    for (int i = 0; i < N; i++)
    {
        numbers[i].i = i;
        numbers[i].j = -i;
        printf("%d %d \n", numbers[i].i, numbers[i].j);
    }
    two_numbers *d_numbers;
    CUDA_ERROR(cudaMalloc((void**)&(d_numbers), N*sizeof(two_numbers)));
    printf("d_i mallocated\n");
    CUDA_ERROR(cudaMemcpy(d_numbers, numbers, N*sizeof(two_numbers), cudaMemcpyHostToDevice));

    printf("d_i memcpied\n");
    increment<<<1,16>>>(d_numbers, N);
    CUDA_ERROR(cudaMemcpy(numbers, d_numbers, N*sizeof(two_numbers), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
    {
        printf("%d %d \n", numbers[i].i, numbers[i].j);
    }
}
