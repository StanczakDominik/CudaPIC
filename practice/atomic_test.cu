#include <stdio.h>

__global__ void addtoall(int* a, int b)
{
    int i = threadIdx.x;
    atomicAdd(&(a[i]), b);
}

int main(void)
{
    int N = 32;
    int *A = new int[N];
    int *d_A;
    cudaMalloc((void**)&d_A, N*sizeof(int));
    cudaMemcpy(d_A, A, N*sizeof(int), cudaMemcpyHostToDevice);
    addtoall<<<1,N>>>(d_A, 7);
    addtoall<<<1,N>>>(d_A, 3);
    addtoall<<<1,N>>>(d_A, 3);
    addtoall<<<1,N>>>(d_A, 3);
    addtoall<<<1,N>>>(d_A, 3);
    addtoall<<<1,N>>>(d_A, 3);
    addtoall<<<1,N>>>(d_A, 3);
    cudaMemcpy(A, d_A, N*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i =0; i<N; i++)
    {
        printf("%d ", A[i]);
    }
}
