#include "helpers.cuh"

void CUDA_ERROR( cudaError_t err){
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s, exiting\n", cudaGetErrorString(err));
        exit(-1);
    }
}
