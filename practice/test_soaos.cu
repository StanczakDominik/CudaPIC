#include <stdio.h>


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

struct container
{
    two_numbers *numbers;
    two_numbers *d_numbers;
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

    two_numbers *numbers = new two_numbers[N];
    container theBox;
    theBox.numbers = numbers;

    for (int i = 0; i < N; i++)
    {
        theBox.numbers[i].i = i;
        theBox.numbers[i].j = -i;
        printf("%d %d \n", theBox.numbers[i].i, theBox.numbers[i].j);
    }
    two_numbers *d_numbers;
    theBox.d_numbers = d_numbers;
    CUDA_ERROR(cudaMalloc((void**)&(theBox.d_numbers), N*sizeof(two_numbers))); //this causes cudaMemcpy

    printf("d_i mallocated\n");
    // CUDA_ERROR(cudaMemcpy(d_numbers, numbers, N*sizeof(two_numbers), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(theBox.d_numbers, theBox.numbers, N*sizeof(two_numbers), cudaMemcpyHostToDevice));

    printf("d_i memcpied\n");
    increment<<<1,16>>>(theBox.d_numbers, N);
    CUDA_ERROR(cudaMemcpy(theBox.numbers, theBox.d_numbers, N*sizeof(two_numbers), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
    {
        printf("%d %d \n", theBox.numbers[i].i, theBox.numbers[i].j);
    }
    CUDA_ERROR(cudaFree(theBox.d_numbers));
}
