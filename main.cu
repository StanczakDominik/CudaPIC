#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <iostream>
using namespace std;

#define ELECTRON_MASS 9.10938356e-31
#define PROTON_MASS 1.6726219e-27
#define ELECTRON_CHARGE 1.6021765e-19
#define EPSILON_ZERO 8.854e-12

#define N_particles_1_axis 64 //should be maybe connected to threadsPerBlock somehow
#define N_particles  (N_particles_1_axis*N_particles_1_axis*N_particles_1_axis) //does this compile with const? //2^4^3 = 2^7 = 128
#define L 1e-4
#define dt 1e-11
//TODO: THIS MAY HAVE TO BE CORRECTED
#define NT 1
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


dim3 particleThreads(32);
dim3 particleBlocks(N_particles/particleThreads.x);
dim3 gridThreads(16,16,16);
dim3 gridBlocks(N_grid/gridThreads.x, N_grid/gridThreads.y, N_grid/gridThreads.z);
static void CUDA_ERROR( cudaError_t err)
{
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s, exiting\n", cudaGetErrorString(err));
        exit(-1);
    }
}
struct Grid{
    // int N_grid;
    // int N_grid_all;
    // float dx;

    float *rho;
    float *Ex;
    float *Ey;
    float *Ez;

    float *d_rho;
    float *d_Ex;
    float *d_Ey;
    float *d_Ez;
    //fourier transformed versions of grid quantities, for fields solver
    cufftComplex *d_fourier_rho;
    cufftComplex *d_fourier_Ex;
    cufftComplex *d_fourier_Ey;
    cufftComplex *d_fourier_Ez;

    cufftHandle plan_forward;
    cufftHandle plan_backward;

    float *kv;
    float *d_kv; //wave vector for field solver
};

struct Particle{
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

struct Species{
    float m;
    float q;
    long int N;

    Particle *particles;
    Particle *d_particles;
};

__global__ void solve_poisson(float *d_kv, cufftComplex *d_fourier_rho, cufftComplex *d_fourier_Ex, cufftComplex *d_fourier_Ey, cufftComplex *d_fourier_Ez){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int index = k*N_grid*N_grid + j*N_grid + i*N_grid;
    if(i<N_grid && j<N_grid && k<N_grid){
        float k2 = d_kv[i]*d_kv[i] + d_kv[j]*d_kv[j] + d_kv[k]*d_kv[k];
        if (i==0 && j==0 && k ==0)    {
            k2 = 1.0f;
        }

        //see: birdsall langdon page 19
        d_fourier_Ex[index].x = -d_kv[i]*d_fourier_rho[index].x/k2/EPSILON_ZERO;
        d_fourier_Ex[index].y = -d_kv[i]*d_fourier_rho[index].y/k2/EPSILON_ZERO;

        d_fourier_Ey[index].x = -d_kv[j]*d_fourier_rho[index].x/k2/EPSILON_ZERO;
        d_fourier_Ey[index].y = -d_kv[j]*d_fourier_rho[index].y/k2/EPSILON_ZERO;

        d_fourier_Ez[index].x = -d_kv[k]*d_fourier_rho[index].x/k2/EPSILON_ZERO;
        d_fourier_Ez[index].y = -d_kv[k]*d_fourier_rho[index].y/k2/EPSILON_ZERO;
    }
}

__global__ void real2complex(float *input, cufftComplex *output){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int index = k*N_grid*N_grid + j*N_grid + i*N_grid;

    if(i<N_grid && j<N_grid && k<N_grid)    {
        output[index].x = input[index];
        output[index].y = 0.0f;
    }
}
__global__ void complex2real(cufftComplex *input, float *output){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int index = k*N_grid*N_grid + j*N_grid + i*N_grid;

    if(i<N_grid && j<N_grid && k<N_grid){
        output[index] = input[index].x/float(N_grid_all);
    }
}

__global__ void scale_down_after_fft(float *d_Ex, float *d_Ey, float *d_Ez){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int index = k*N_grid*N_grid + j*N_grid + i*N_grid;

    if(i<N_grid && j<N_grid && k<N_grid){
        d_Ex[index] /= float(N_grid_all);
        d_Ey[index] /= float(N_grid_all);
        d_Ez[index] /= float(N_grid_all);
    }
}

__global__ void set_grid_array_to_value(float *arr, float value){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int index = k*N_grid*N_grid + j*N_grid + i*N_grid;

    if(i<N_grid && j<N_grid && k<N_grid){
        arr[index] = value;
    }
}

void init_grid(Grid *g){
    g->rho = new float[N_grid_all];
    g->Ex = new float[N_grid_all];
    g->Ey = new float[N_grid_all];
    g->Ez = new float[N_grid_all];

    g->kv = new float[N_grid];
    for (int i =0; i<=N_grid/2; i++)
    {
        g->kv[i] = i*2*M_PI;
    }
    for (int i = N_grid/2 + 1; i < N_grid; i++)
    {
        g->kv[i] = (i-N_grid)*2*M_PI;
    }


    CUDA_ERROR(cudaMalloc((void**)&(g->d_kv), sizeof(float)*N_grid));
    CUDA_ERROR(cudaMemcpy(g->d_kv, g->kv, sizeof(float)*N_grid, cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMalloc((void**)&(g->d_fourier_rho), sizeof(cufftComplex)*N_grid_all));
    CUDA_ERROR(cudaMalloc((void**)&(g->d_fourier_Ex), sizeof(cufftComplex)*N_grid_all));
    CUDA_ERROR(cudaMalloc((void**)&(g->d_fourier_Ey), sizeof(cufftComplex)*N_grid_all));
    CUDA_ERROR(cudaMalloc((void**)&(g->d_fourier_Ez), sizeof(cufftComplex)*N_grid_all));
    CUDA_ERROR(cudaMalloc((void**)&(g->d_rho), sizeof(float)*N_grid_all));
    CUDA_ERROR(cudaMemcpy(g->d_rho, g->rho, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMalloc((void**)&(g->d_Ex), sizeof(float)*N_grid_all));
    CUDA_ERROR(cudaMemcpy(g->d_Ex, g->Ex, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMalloc((void**)&(g->d_Ey), sizeof(float)*N_grid_all));
    CUDA_ERROR(cudaMemcpy(g->d_Ey, g->Ey, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMalloc((void**)&(g->d_Ez), sizeof(float)*N_grid_all));
    CUDA_ERROR(cudaMemcpy(g->d_Ez, g->Ez, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice));
    cufftPlan3d(&(g->plan_forward), N_grid, N_grid, N_grid, CUFFT_R2C);
    cufftPlan3d(&(g->plan_backward), N_grid, N_grid, N_grid, CUFFT_C2R);
}


void debug_field_solver_uniform(Grid *g){
    float* linear_field_x = new float[N_grid_all];
    float* linear_field_y = new float[N_grid_all];
    float* linear_field_z = new float[N_grid_all];
    for(int i = 0; i<N_grid;  i++){
        for(int j = 0; j<N_grid;  j++){
            for(int k = 0; k<N_grid;  k++){
                int index = k*N_grid*N_grid + j*N_grid + i;
                linear_field_x[index] = 1;
                linear_field_y[index] = 0;
                linear_field_z[index] = 0;
            }
        }
    }
    cudaMemcpy(g->d_Ex, linear_field_x, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_Ey, linear_field_y, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_Ez, linear_field_z, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice);
}
void debug_field_solver_linear(Grid *g)
{
    float* linear_field_x = new float[N_grid_all];
    float* linear_field_y = new float[N_grid_all];
    float* linear_field_z = new float[N_grid_all];
    for(int i = 0; i<N_grid;  i++){
        for(int j = 0; j<N_grid;  j++){
            for(int k = 0; k<N_grid;  k++){
                int index = k*N_grid*N_grid + j*N_grid + i;
                linear_field_x[index] = dx*i;
                linear_field_y[index] = dx*j;
                linear_field_z[index] = dx*k;
            }
        }
    }
    cudaMemcpy(g->d_Ex, linear_field_x, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_Ey, linear_field_y, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_Ez, linear_field_z, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice);
}
void debug_field_solver_quadratic(Grid *g)
{
    float* linear_field_x = new float[N_grid_all];
    float* linear_field_y = new float[N_grid_all];
    float* linear_field_z = new float[N_grid_all];
    for(int i = 0; i<N_grid;  i++){
        for(int j = 0; j<N_grid;  j++){
            for(int k = 0; k<N_grid;  k++){
                int index = k*N_grid*N_grid + j*N_grid + i;
                linear_field_x[index] = (dx*i)*(dx*i);
                linear_field_y[index] = (dx*j)*(dx*j);
                linear_field_z[index] = (dx*k)*(dx*k);
            }
        }
    }
    cudaMemcpy(g->d_Ex, linear_field_x, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_Ey, linear_field_y, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_Ez, linear_field_z, sizeof(float)*N_grid_all, cudaMemcpyHostToDevice);
}

void field_solver(Grid *g){
    cufftExecR2C(g->plan_forward, g->d_rho, g->d_fourier_rho);

    solve_poisson<<<gridBlocks, gridThreads>>>(g->d_kv, g->d_fourier_rho, g->d_fourier_Ex, g->d_fourier_Ey, g->d_fourier_Ez);
    cufftExecC2R(g->plan_backward, g->d_fourier_Ex, g->d_Ex);
    cufftExecC2R(g->plan_backward, g->d_fourier_Ey, g->d_Ey);
    cufftExecC2R(g->plan_backward, g->d_fourier_Ez, g->d_Ez);

    scale_down_after_fft<<<gridBlocks, gridThreads>>>(g->d_Ex, g->d_Ey, g->d_Ez);
}

__device__ int position_to_grid_index(float X){
    return int(X/dx);
}
__device__ float position_in_cell(float x){
    int grid_index = position_to_grid_index(x);
    return x - grid_index*dx;
}

__global__ void scatter_charge(Particle *d_P, float q, float* d_rho){
    int n = blockIdx.x*blockDim.x + threadIdx.x;

    float x = d_P[n].x;
    float y = d_P[n].y;
    float z = d_P[n].z;
    int i = position_to_grid_index(x);
    int j = position_to_grid_index(y);
    int k = position_to_grid_index(z);

    float Xr = position_in_cell(x)/dx;
    float Xl = 1 - Xr;
    float Yr = position_in_cell(y)/dy;
    float Yl = 1 - Yr;
    float Zr = position_in_cell(z)/dz;
    float Zl = 1 - Zr;

    //this part is literally hitler - not just unreadable but slow af
    //TODO: redo this using a reduce, maybe?
    atomicAdd(&(d_rho[N_grid * N_grid * (k)%N_grid + N_grid * (j)%N_grid + (i)%N_grid]), q*Xl*Yl*Zl);
    atomicAdd(&(d_rho[N_grid * N_grid * (k)%N_grid + N_grid * (j)%N_grid + (i+1)%N_grid]), q*Xr*Yl*Zl);
    atomicAdd(&(d_rho[N_grid * N_grid * (k)%N_grid + N_grid * (j+1)%N_grid + (i)%N_grid]), q*Xl*Yr*Zl);
    atomicAdd(&(d_rho[N_grid * N_grid * (k+1)%N_grid + N_grid * (j)%N_grid + (i)%N_grid]), q*Xl*Yl*Zr);
    atomicAdd(&(d_rho[N_grid * N_grid * (k)%N_grid + N_grid * (j+1)%N_grid + (i+1)%N_grid]), q*Xr*Yr*Zl);
    atomicAdd(&(d_rho[N_grid * N_grid * (k+1)%N_grid + N_grid * (j)%N_grid + (i+1)%N_grid]), q*Xr*Yl*Zr);
    atomicAdd(&(d_rho[N_grid * N_grid * (k+1)%N_grid + N_grid * (j+1)%N_grid + (i)%N_grid]), q*Xl*Yr*Zr);
    atomicAdd(&(d_rho[N_grid * N_grid * (k+1)%N_grid + N_grid * (j+1)%N_grid + (i+1)%N_grid]), q*Xr*Yr*Zr);
}
__device__ float gather_grid_to_particle(Particle *p, float *grid){
    float x = p->x;
    float y = p->y;
    float z = p->z;
    int i = position_to_grid_index(x);
    int j = position_to_grid_index(y);
    int k = position_to_grid_index(z);

    float Xr = position_in_cell(x)/dx;
    float Xl = 1 - Xr;
    float Yr = position_in_cell(y)/dy;
    float Yl = 1 - Yr;
    float Zr = position_in_cell(z)/dz;
    float Zl = 1 - Zr;

    float interpolated_scalar = 0.0f;
    //this part is also hitler but not as much
    interpolated_scalar += grid[N_grid * N_grid * (k)%N_grid + N_grid * (j)%N_grid + (i)%N_grid]*Xl*Yl*Zl;
    interpolated_scalar += grid[N_grid * N_grid * (k)%N_grid + N_grid * (j)%N_grid + (i+1)%N_grid]*Xr*Yl*Zl;
    interpolated_scalar += grid[N_grid * N_grid * (k)%N_grid + N_grid * (j+1)%N_grid + (i)%N_grid]*Xl*Yr*Zl;
    interpolated_scalar += grid[N_grid * N_grid * (k+1)%N_grid + N_grid * (j)%N_grid + (i)%N_grid]*Xl*Yl*Zr;
    interpolated_scalar += grid[N_grid * N_grid * (k)%N_grid + N_grid * (j+1)%N_grid + (i+1)%N_grid]*Xr*Yr*Zl;
    interpolated_scalar += grid[N_grid * N_grid * (k+1)%N_grid + N_grid * (j)%N_grid + (i+1)%N_grid]*Xr*Yl*Zr;
    interpolated_scalar += grid[N_grid * N_grid * (k+1)%N_grid + N_grid * (j+1)%N_grid + (i)%N_grid]*Xl*Yr*Zr;
    interpolated_scalar += grid[N_grid * N_grid * (k+1)%N_grid + N_grid * (j+1)%N_grid + (i+1)%N_grid]*Xr*Yr*Zr;
    return interpolated_scalar;

}


__global__ void InitParticleArrays(Particle *d_p, float shiftx, float shifty, float shiftz){
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n<N_particles){
        Particle *p = &(d_p[n]);

        int i = n / (int)(N_particles_1_axis*N_particles_1_axis);
        int j = (int) (n/N_particles_1_axis) % N_particles_1_axis;
        int k = n % N_particles_1_axis;
        p->x = L/float(N_particles_1_axis) * i + shiftx;
        p->x = p->x - floor(p->x/L)*L;
        p->y = L/float(N_particles_1_axis) * j + shifty;
        p->y = p->y - floor(p->y/L)*L;
        p->z = L/float(N_particles_1_axis) * k + shiftz;
        p->z = p->z - floor(p->z/L)*L;

        p->vx = 0.0f;
        p->vy = 0.0f;
        p->vz = 0.0f;
    }
}
__global__ void InitialVelocityStep(Particle *d_p, float q, float m, float *d_Ex, float *d_Ey, float *d_Ez){
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    {
        Particle *p = &(d_p[n]);
        //gather electric field
        float Ex = gather_grid_to_particle(p, d_Ex);
        float Ey = gather_grid_to_particle(p, d_Ey);
        float Ez = gather_grid_to_particle(p, d_Ez);

       //use electric field to accelerate particles
       p->vx -= 0.5f*dt*q/m*Ex;
       p->vy -= 0.5f*dt*q/m*Ey;
       p->vz -= 0.5f*dt*q/m*Ez;
    }
}

__global__ void ParticleKernel(Particle *d_p, float q, float m, float *d_Ex, float *d_Ey, float *d_Ez){
   int n = blockDim.x * blockIdx.x + threadIdx.x;
   if(n<N_particles)
   {
       Particle *p = &(d_p[n]);
       //push positions, enforce periodic boundary conditions

       p->x = p->x + p->vx*dt;
       p->x = p->x - floor(p->x/L)*L;

       p->y = p->y + p->vy*dt;
       p->y = p->y - floor(p->y/L)*L;

       p->z = p->z + p->vz*dt;
       p->z = p->z - floor(p->z/L)*L;

       //gather electric field
       float Ex = gather_grid_to_particle(p, d_Ex);
       float Ey = gather_grid_to_particle(p, d_Ey);
       float Ez = gather_grid_to_particle(p, d_Ez);

       //use electric field to accelerate particles
       p->vx += dt*q/m*Ex;
       p->vy += dt*q/m*Ey;
       p->vz += dt*q/m*Ez;
   }
}


void init_species(Species *s, float shiftx, float shifty, float shiftz){
    s->particles = new Particle[N_particles];
    CUDA_ERROR(cudaMalloc((void**)&(s->d_particles), sizeof(Particle)*N_particles));
    cout << "initializing particles" << endl;
    InitParticleArrays<<<particleBlocks, particleThreads>>>(s->d_particles, shiftx, shifty, shiftz);
}

void dump_density_data(Grid *g, char* name){
    cout << "dumping" << endl;
    CUDA_ERROR(cudaMemcpy(g->rho, g->d_rho, sizeof(float)*N_grid_all, cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(g->Ex, g->d_Ex, sizeof(float)*N_grid_all, cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(g->Ey, g->d_Ey, sizeof(float)*N_grid_all, cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(g->Ez, g->d_Ez, sizeof(float)*N_grid_all, cudaMemcpyDeviceToHost));
    FILE *density_data = fopen(name, "w");
    for (int n = 0; n < N_grid_all; n++)
    {
        fprintf(density_data, "%f %f %f %f\n", g->rho[n], g->Ex[n], g->Ey[n], g->Ez[n]);
    }
    for(int n = 0; n < 10; n++)
    {
        printf("%f %f %f %f\n", g->rho[n], g->Ex[n], g->Ey[n], g->Ez[n]);
    }
    // free(g->rho);
}

void dump_position_data(Species *s, char* name){
    cout << "Copying particles from GPU to device"<< endl;
    CUDA_ERROR(cudaMemcpy(s->particles, s->d_particles, sizeof(Particle)*N_particles, cudaMemcpyDeviceToHost));
    cout << "Copied particles from GPU to device"<< endl;
    FILE *initial_position_data = fopen(name, "w");
    for (int i =0; i<N_particles; i++)
    {
        Particle *p = &(s->particles[i]);
        fprintf(initial_position_data, "%f %f %f %f %f %f\n", p->x, p->y, p->z, p->vx, p->vy, p->vz);
    }
    // free(s->particles);
}

void init_timestep(Grid *g, Species *electrons,  Species *ions){
    set_grid_array_to_value<<<gridBlocks, gridThreads>>>(g->d_rho, 0);
    CUDA_ERROR(cudaDeviceSynchronize());
    scatter_charge<<<particleBlocks, particleThreads>>>(electrons->d_particles, electrons->q, g->d_rho);
    scatter_charge<<<particleBlocks, particleThreads>>>(ions->d_particles, ions->q, g->d_rho);
    CUDA_ERROR(cudaDeviceSynchronize());

    debug_field_solver_uniform(g);
    // field_solver(g);
    CUDA_ERROR(cudaDeviceSynchronize());

    InitialVelocityStep<<<particleBlocks, particleThreads>>>(electrons->d_particles, electrons->q, electrons->m, g->d_Ex, g->d_Ey, g->d_Ez);
    InitialVelocityStep<<<particleBlocks, particleThreads>>>(ions->d_particles, ions->q, ions->m, g->d_Ex, g->d_Ey, g->d_Ez);
    CUDA_ERROR(cudaDeviceSynchronize());
}


void timestep(Grid *g, Species *electrons,  Species *ions){
	//1. move particles, gather electric fields at their locations, accelerate particles
	ParticleKernel<<<particleBlocks, particleThreads>>>(electrons->d_particles, electrons->q, electrons->m, g->d_Ex, g->d_Ey, g->d_Ez);
	ParticleKernel<<<particleBlocks, particleThreads>>>(ions->d_particles, ions->q, ions->m, g->d_Ex, g->d_Ey, g->d_Ez);
	//potential TODO: sort particles?????
    //2. clear charge density for scattering fields to particles charge
    set_grid_array_to_value<<<gridBlocks, gridThreads>>>(g->d_rho, 0);
    CUDA_ERROR(cudaDeviceSynchronize());


    //3. gather charge from new particle position to grid
    //TODO: note that I may need to cudaSyncThreads between these steps
    scatter_charge<<<particleBlocks, particleThreads>>>(electrons->d_particles, electrons->q, g->d_rho);
    scatter_charge<<<particleBlocks, particleThreads>>>(ions->d_particles, ions->q, g->d_rho);
    CUDA_ERROR(cudaDeviceSynchronize());

    //4. use charge density to calculate field
    debug_field_solver_uniform(g);
    // field_solver(g);
    CUDA_ERROR(cudaDeviceSynchronize());
}

int main(void){
    Grid g;
    init_grid(&g);
    dump_density_data(&g, "initial_density.dat");

    Species electrons;
    electrons.q = -ELECTRON_CHARGE;
    electrons.m = ELECTRON_MASS;
    electrons.N = N_particles;
    init_species(&electrons, L/100.0f, 0, 0);

    Species ions;
    ions.q = +ELECTRON_CHARGE;
    ions.m = PROTON_MASS;
    ions.N = N_particles;
    init_species(&ions, 0, 0, 0);

    init_timestep(&g, &electrons, &ions);

    dump_position_data(&ions, "ions_positions.dat");
    dump_position_data(&electrons, "electrons_positions.dat.");
    dump_density_data(&g, "initial_density.dat");
    cout << "entering time loop" << endl;
    for(int i =0; i<NT; i++){
        timestep(&g, &electrons, &ions);
    }
    cout << "finished time loop" << endl;

    dump_position_data(&ions, "final_ions_positions.dat");
    dump_position_data(&electrons, "final_electrons_positions.dat.");
    dump_density_data(&g, "final_density.dat");
    CUDA_ERROR(cudaFree(electrons.d_particles));
    CUDA_ERROR(cudaFree(g.d_rho));
    CUDA_ERROR(cudaFree(g.d_Ex));
    CUDA_ERROR(cudaFree(g.d_Ey));
    CUDA_ERROR(cudaFree(g.d_Ez));
    CUDA_ERROR(cudaFree(g.d_fourier_Ex));
    CUDA_ERROR(cudaFree(g.d_fourier_Ey));
    CUDA_ERROR(cudaFree(g.d_fourier_Ez));
    CUDA_ERROR(cudaFree(g.d_fourier_rho));
}
