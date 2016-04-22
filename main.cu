#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>


#define N_particles_1_axis 128 //should be maybe connected to threadsPerBlock somehow
#define N_particles  (N_particles_1_axis*N_particles_1_axis*N_particles_1_axis) //does this compile with const? //2^4^3 = 2^7 = 128
#define L 1.f
#
#define dt 0.01f
#define NT 500
#define N_grid 16
#define N_grid_all (N_grid *N_grid * N_grid)
#define dx (L/float(N_grid))
#define dy dx
#define dz dx

size_t particle_array_size = N_particles*sizeof(float);
// size_t grid_array_size = N_grid*sizeof(float);

/*
Assumptions:
q=1
m=1
L = 1
*/


dim3 particleThreads(64);
dim3 particleBlocks(N_particles/particleThreads.x);
dim3 gridThreads(16,16,16);
dim3 gridBlocks(N_grid/gridThreads.x, N_grid/gridThreads.y, N_grid/gridThreads.z);

// __global__ void solve_poisson()
// {
//     int i = blockIdx.x*blockDim.x + threadIdx.x;
//     int j = blockIdx.y*blockDim.y + threadIdx.y;
//     int k = blockIdx.z*blockDim.z + threadIdx.z;
//
//     int index = k*N_grid*N_grid + j*N_grid + i*N_grid;
//     if(i<N_grid && j<N_grid && k<N_grid)
//     {
//         float k2 = kv[i]*kv[i] + kv[j]*kv[j] + kv[k]*kv[k];
//         if (i==0 && j==0 && k ==0)
//             k2 = 1.0f;
//         }
//     }
// }


__device__ int position_to_grid_index(float X)
{
    return int(X/dx);
}

__device__ float position_in_cell(float x)
{
    int grid_index = position_to_grid_index(x);
    return x - grid_index*dx;
}


__global__ void scatter_charge(float* X, float* Y, float* Z, float* d_charge, float q)
{
    int n = blockIdx.x*blockDim.x + threadIdx.x;

    int i = position_to_grid_index(X[n]);
    int j = position_to_grid_index(Y[n]);
    int k = position_to_grid_index(Z[n]);

    float Xr = position_in_cell(X[n])/dx;
    float Xl = 1 - Xr;
    float Yr = position_in_cell(Y[n])/dy;
    float Yl = 1 - Yr;
    float Zr = position_in_cell(Z[n])/dz;
    float Zl = 1 - Zr;

    //this part is literally hitler - not just unreadable but slow af
    //TODO: redo this using a reduce, maybe?
    atomicAdd(&(d_charge[N_grid * N_grid * (k)%N_grid + N_grid * (j)%N_grid + (i)%N_grid]), q*Xl*Yl*Zl);
    atomicAdd(&(d_charge[N_grid * N_grid * (k)%N_grid + N_grid * (j)%N_grid + (i+1)%N_grid]), q*Xr*Yl*Zl);
    atomicAdd(&(d_charge[N_grid * N_grid * (k)%N_grid + N_grid * (j+1)%N_grid + (i)%N_grid]), q*Xl*Yr*Zl);
    atomicAdd(&(d_charge[N_grid * N_grid * (k+1)%N_grid + N_grid * (j)%N_grid + (i)%N_grid]), q*Xl*Yl*Zr);
    atomicAdd(&(d_charge[N_grid * N_grid * (k)%N_grid + N_grid * (j+1)%N_grid + (i+1)%N_grid]), q*Xr*Yr*Zl);
    atomicAdd(&(d_charge[N_grid * N_grid * (k+1)%N_grid + N_grid * (j)%N_grid + (i+1)%N_grid]), q*Xr*Yl*Zr);
    atomicAdd(&(d_charge[N_grid * N_grid * (k+1)%N_grid + N_grid * (j+1)%N_grid + (i)%N_grid]), q*Xl*Yr*Zr);
    atomicAdd(&(d_charge[N_grid * N_grid * (k+1)%N_grid + N_grid * (j+1)%N_grid + (i+1)%N_grid]), q*Xr*Yr*Zr);
}


__device__ float gather_grid_to_particle(float x, float y, float z, float *grid)
{
    int i = position_to_grid_index(x);
    int j = position_to_grid_index(y);
    int k = position_to_grid_index(z);

    float Xr = position_in_cell(x)/dx;
    float Xl = 1 - Xr;
    float Yr = position_in_cell(y)/dy;
    float Yl = 1 - Yr;
    float Zr = position_in_cell(z)/dz;
    float Zl = 1 - Zr;

    float result = 0.0f;
    //this part is also hitler but not as much
    result += grid[N_grid * N_grid * (k)%N_grid + N_grid * (j)%N_grid + (i)%N_grid]*Xl*Yl*Zl;
    result += grid[N_grid * N_grid * (k)%N_grid + N_grid * (j)%N_grid + (i+1)%N_grid]*Xr*Yl*Zl;
    result += grid[N_grid * N_grid * (k)%N_grid + N_grid * (j+1)%N_grid + (i)%N_grid]*Xl*Yr*Zl;
    result += grid[N_grid * N_grid * (k+1)%N_grid + N_grid * (j)%N_grid + (i)%N_grid]*Xl*Yl*Zr;
    result += grid[N_grid * N_grid * (k)%N_grid + N_grid * (j+1)%N_grid + (i+1)%N_grid]*Xr*Yr*Zl;
    result += grid[N_grid * N_grid * (k+1)%N_grid + N_grid * (j)%N_grid + (i+1)%N_grid]*Xr*Yl*Zr;
    result += grid[N_grid * N_grid * (k+1)%N_grid + N_grid * (j+1)%N_grid + (i)%N_grid]*Xl*Yr*Zr;
    result += grid[N_grid * N_grid * (k+1)%N_grid + N_grid * (j+1)%N_grid + (i+1)%N_grid]*Xr*Yr*Zr;
    return result;

}



__global__ void InitParticleArrays(float* X, float *Y, float* Z, float *Vx, float *Vy, float *Vz)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n<N_particles)
    {
        X[n] = L/float(N_particles_1_axis)*(n%N_particles_1_axis);
        Y[n] = L/float(N_particles_1_axis)*(n/N_particles_1_axis)/float(N_particles_1_axis);
        Z[n] = L/float(N_particles_1_axis)*(n/N_particles_1_axis/N_particles_1_axis);
        Vx[n] = 0.1f;
        Vy[n] = 0.2f;
        Vz[n] = 0.3f;
    }
}

__global__ void InitialVelocityStep(float* X, float* Y, float* Z, float* Vx, float* Vy, float* Vz)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    {
        Vx[n] -= 0.5f*dt*(-2*(X[n]-0.5f));
        Vy[n] -= 0.5f*dt*(-2*(Y[n]-0.5f));
        Vz[n] -= 0.5f*dt*(-2*(Z[n]-0.5f));
    }
}


__global__ void ParticleKernel(float *X, float *Y, float *Z, float *Vx, float *Vy, float *Vz)//, float *d_Ex, float *d_Ey, float *d_Ez, float qm)
{
   int n = blockDim.x * blockIdx.x + threadIdx.x;
   if(n<N_particles)
   {
       //push positions, enforce periodic boundary conditions
       X[n] = fmod((X[n] + Vx[n]*dt),L);
       Y[n] = fmod((Y[n] + Vy[n]*dt),L);
       Z[n] = fmod((Z[n] + Vz[n]*dt),L);
       //gather electric field
    //    float Ex = gather_grid_to_particle(X[n], Y[n], Z[n], d_Ex);
    //    float Ey = gather_grid_to_particle(X[n], Y[n], Z[n], d_Ey);
    //    float Ez = gather_grid_to_particle(X[n], Y[n], Z[n], d_Ez);

       //use electric field to accelerate particles
       Vx[n] += dt*(-2*(X[n]-0.5f));
       Vy[n] += dt*(-2*(Y[n]-0.5f));
       Vz[n] += dt*(-2*(Z[n]-0.5f));
   }
}

int main(void)
{
    //TODO: routine checks for cuda status

    float* d_X;
    float* d_Y;
    float* d_Z;
    float* d_Vx;
    float* d_Vy;
    float* d_Vz;

    float *X = new float[N_particles];
    float *Y = new float[N_particles];
    float *Z = new float[N_particles];
    float *Vx = new float[N_particles];
    float *Vy = new float[N_particles];
    float *Vz = new float[N_particles];


    cudaMalloc((void**)&d_X, sizeof(float)*N_particles);
    cudaMalloc((void**)&d_Y, sizeof(float)*N_particles);
    cudaMalloc((void**)&d_Z, sizeof(float)*N_particles);
    cudaMalloc((void**)&d_Vx, sizeof(float)*N_particles);
    cudaMalloc((void**)&d_Vy, sizeof(float)*N_particles);
    cudaMalloc((void**)&d_Vz, sizeof(float)*N_particles);
    InitParticleArrays<<<particleBlocks, particleThreads>>>(d_X, d_Y, d_Z, d_Vx, d_Vy, d_Vz);
    InitialVelocityStep<<<particleBlocks, particleThreads>>>(d_X, d_Y, d_Z, d_Vx, d_Vy, d_Vz);

    float *charge = new float[N_grid_all];
    float *d_charge = new float[N_grid_all];

    cudaMalloc((void**)&d_charge, sizeof(float)*N_grid_all);
    cudaMemset(&d_charge, sizeof(float)*N_grid_all, 0);
    scatter_charge<<<particleBlocks, particleThreads>>>(d_X, d_Y, d_Z, d_charge, 1);


    cudaMemcpy(charge, d_charge, sizeof(float)*N_grid_all, cudaMemcpyDeviceToHost);
    FILE *density_data = fopen("density_data.dat", "w");
    for (int n = 0; n < N_grid_all; n++)
    {
        fprintf(density_data, "%f\n", charge[n]);
    }


    cudaMemcpy(X, d_X, sizeof(float)*N_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(Y, d_Y, sizeof(float)*N_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(Z, d_Z, sizeof(float)*N_particles, cudaMemcpyDeviceToHost);

    FILE *initial_position_data = fopen("initial.dat", "w");
    for (int p =0; p<N_particles; p++)
    {
        fprintf(initial_position_data, "%f %f %f\n", X[p], Y[p], Z[p]);
    }

    FILE *trajectory_data = fopen("trajectory.dat", "w");
    for(int i =0; i<NT; i++)
    {
        ParticleKernel<<<particleBlocks, particleThreads>>>(d_X, d_Y, d_Z, d_Vx, d_Vy, d_Vz);
        cudaMemcpy(X, d_X, sizeof(float)*N_particles, cudaMemcpyDeviceToHost);
        cudaMemcpy(Y, d_Y, sizeof(float)*N_particles, cudaMemcpyDeviceToHost);
        cudaMemcpy(Z, d_Z, sizeof(float)*N_particles, cudaMemcpyDeviceToHost);
        for (int p =0; p<10; p++)
        {
            fprintf(trajectory_data,"%f %f %f ", X[p], Y[p], Z[p]);
        }
        fprintf(trajectory_data, "\n");
    }



    cudaMemcpy(X, d_X, sizeof(float)*N_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(Y, d_Y, sizeof(float)*N_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(Z, d_Z, sizeof(float)*N_particles, cudaMemcpyDeviceToHost);
    FILE *final_position_data = fopen("final.dat", "w");
    for (int p =0; p<N_particles; p++)
    {
        fprintf(final_position_data, "%f %f %f\n", X[p], Y[p], Z[p]);
    }

    cudaMemset(&d_charge, sizeof(float)*N_grid_all, 0);
    scatter_charge<<<particleBlocks, particleThreads>>>(d_X, d_Y, d_Z, d_charge, 1);
    cudaMemcpy(charge, d_charge, sizeof(float)*N_grid_all, cudaMemcpyDeviceToHost);
    FILE *final_density_data = fopen("final_density_data.dat", "w");
    for (int n = 0; n < N_grid_all; n++)
    {
        fprintf(final_density_data, "%f\n", charge[n]);
    }

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);
    cudaFree(d_Vx);
    cudaFree(d_Vy);
    cudaFree(d_Vz);
    cudaFree(d_charge);
}
