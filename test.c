#include <stdio.h>


__global__ add(int* d_A, int* d_B)
{
  
}
int main(){
  #define BLOCK_SIZE 16
  #define GRID_SIZE 1
  int d_A[BLOCK_SIZE][BLOCK_SIZE];
  int d_B[BLOCK_SIZE][BLOCK_SIZE];

  /* d_A initialization */

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
  dim3 dimGrid(GRID_SIZE, GRID_SIZE); // 1*1 blocks in a grid

  YourKernel<<<dimGrid, dimBlock>>>(d_A,d_B); //Kernel invocation
}
