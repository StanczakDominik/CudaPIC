#include <stdio.h>
/*CUDA error wraper*/
static void CUDA_ERROR( cudaError_t err)
{
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s, exiting\n", cudaGetErrorString(err));
        exit(-1);
    }
}


struct Arrays
{
    int* i;
    int* j;
    int N;
};

__global__ void increment(Arrays d_arrays)
{

    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<d_arrays.N)
    {
        d_arrays.i[i] *= 2;
        d_arrays.j[i] *= -3;
    }
}

int main()
{
    int N = 10;
    Arrays arrays;
    arrays.i = new int[N];
    arrays.j = new int[N];
    arrays.N = N;
    for (int i = 0; i < N; i++)
    {
        arrays.i[i] = i;
        arrays.j[i] = -i;
        printf("%d %d \n", arrays.i[i], arrays.j[i]);
    }
    Arrays d_arrays;
    d_arrays.N = N;
    CUDA_ERROR(cudaMalloc((void**)&(d_arrays.i), N*sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&(d_arrays.j), N*sizeof(int)));
    printf("d_i mallocated\n");
    CUDA_ERROR(cudaMemcpy(d_arrays.i, arrays.i, N*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_arrays.j, arrays.j, N*sizeof(int), cudaMemcpyHostToDevice));

    printf("d_i memcpied\n");
    increment<<<1,16>>>(d_arrays);
    CUDA_ERROR(cudaGetLastError());
    printf("incremented\n");
    CUDA_ERROR(cudaMemcpy(arrays.i, d_arrays.i, N*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(arrays.j, d_arrays.j, N*sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
    {
        printf("%d %d \n", arrays.i[i], arrays.j[i]);
    }

    CUDA_ERROR(cudaFree(d_arrays.i));
    CUDA_ERROR(cudaFree(d_arrays.j));
}
