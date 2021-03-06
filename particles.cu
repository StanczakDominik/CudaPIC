#include "particles.cuh"
__device__ int position_to_grid_index(float x, float dx){
    return x/dx;
}
__device__ float position_in_cell(float x, float dx){
    return x - dx * (int)(x/dx);
}
__device__ int ijk_to_n(int i, int j, int k, int N_grid){
    return (N_grid * N_grid * (k%N_grid) + N_grid * (j%N_grid) + (i%N_grid));
}


__global__ void InitParticleArrays(Particle *d_p, float shiftx, float shifty,
        float shiftz, float vx, float vy, float vz, int N_particles_1_axis, int N_particles){
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n<N_particles){
        Particle *p = &(d_p[n]);

        int i = n / (int)(N_particles_1_axis*N_particles_1_axis);
        int j = (int) (n/N_particles_1_axis) % N_particles_1_axis;
        int k = n % N_particles_1_axis;
        p->x = L/float(N_particles_1_axis) * i;
        p->x += shiftx*sin(2*M_PI/L*p->x);
        p->x = p->x - floor(p->x/L)*L;
        p->y = L/float(N_particles_1_axis) * j;
        p->y += shifty*sin(2*M_PI/L*p->y);
        p->y = p->y - floor(p->y/L)*L;
        p->z = L/float(N_particles_1_axis) * k;
        p->z += shiftz*sin(2*M_PI/L*p->z);
        p->z = p->z - floor(p->z/L)*L;

        p->vx = vx;
        p->vy = vy;
        p->vz = vz;

        // if (threadIdx.x == 0)
        // {
        //     printf("%d %f %f %f\n", blockIdx.x, p->x, p->y, p->z);
        // }
    }
}

void init_species(Species *s, float shiftx, float shifty, float shiftz,
        float vx, float vy, float vz,
        int N_particles_1_axis, int N_grid, float dx){
    s->N_particles_1_axis = N_particles_1_axis;
    s->N_particles = N_particles_1_axis * N_particles_1_axis * N_particles_1_axis;
    s->particles = new Particle[s->N_particles];
    s->particleThreads = dim3(pThreads);
    s->particleBlocks = dim3((s->N_particles+s->particleThreads.x - 1)/s->particleThreads.x);

    s->KE = 0;
    s->Px = 0;
    s->Py = 0;
    s->Pz = 0;
    s->block_v2s = new float[s->particleBlocks.x];
    CUDA_ERROR(cudaMalloc((void**)&(s->d_block_v2s), sizeof(float)*s->particleBlocks.x));
    s->block_Px = new float[s->particleBlocks.x];
    CUDA_ERROR(cudaMalloc((void**)&(s->d_block_Px), sizeof(float)*s->particleBlocks.x));
    s->block_Py = new float[s->particleBlocks.x];
    CUDA_ERROR(cudaMalloc((void**)&(s->d_block_Py), sizeof(float)*s->particleBlocks.x));
    s->block_Pz = new float[s->particleBlocks.x];
    CUDA_ERROR(cudaMalloc((void**)&(s->d_block_Pz), sizeof(float)*s->particleBlocks.x));
    CUDA_ERROR(cudaMalloc((void**)&(s->d_sum_Px), sizeof(float)*(s->particleBlocks.x + pThreads - 1)/pThreads));
    CUDA_ERROR(cudaMalloc((void**)&(s->d_sum_Py), sizeof(float)*(s->particleBlocks.x + pThreads - 1)/pThreads));
    CUDA_ERROR(cudaMalloc((void**)&(s->d_sum_Pz), sizeof(float)*(s->particleBlocks.x + pThreads - 1)/pThreads));
    CUDA_ERROR(cudaMalloc((void**)&(s->d_sum_v2s), sizeof(float)*(s->particleBlocks.x + pThreads - 1)/pThreads));
    CUDA_ERROR(cudaMalloc((void**)&(s->d_moments), sizeof(float)*4));
    s->moments = new float[4];

    CUDA_ERROR(cudaMalloc((void**)&(s->d_particles), sizeof(Particle)*s->N_particles));
    printf("initializing particles\n");
    InitParticleArrays<<<s->particleBlocks, s->particleThreads>>>(s->d_particles, shiftx, shifty, shiftz, vx, vy, vz, s->N_particles_1_axis, s->N_particles);
    printf("Blocks: %d %d %d Threads: %d %d %d \n",
        s->particleBlocks.x,
        s->particleBlocks.y,
        s->particleBlocks.z,
        s->particleThreads.x,
        s->particleThreads.y,
        s->particleThreads.z);
    printf("Mass: %f Charge: %f N: %ld\n", s->m, s->q, s->N_particles);

}

__global__ void scatter_charge_kernel(Particle *d_P, float q, float* d_rho, int N_grid, float dx, int N_particles){
    // __shared__ float local_rho[32*32*32];
    int n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n < N_particles){
        float x = d_P[n].x;
        float y = d_P[n].y;
        float z = d_P[n].z;
        int i = position_to_grid_index(x, dx);
        int j = position_to_grid_index(y, dx);
        int k = position_to_grid_index(z, dx);

        float Xr = position_in_cell(x, dx)/dx;
        float Xl = 1 - Xr;
        float Yr = position_in_cell(y, dx)/dx;
        float Yl = 1 - Yr;
        float Zr = position_in_cell(z, dx)/dx;
        float Zl = 1 - Zr;

        //TODO: redo this using a reduce
        atomicAdd(&(d_rho[ijk_to_n(i, j, k, N_grid)]),       q*Xl*Yl*Zl);
        atomicAdd(&(d_rho[ijk_to_n(i+1, j, k, N_grid)]),     q*Xr*Yl*Zl);
        atomicAdd(&(d_rho[ijk_to_n(i, j+1, k, N_grid)]),     q*Xl*Yr*Zl);
        atomicAdd(&(d_rho[ijk_to_n(i, j, k+1, N_grid)]),     q*Xl*Yl*Zr);
        atomicAdd(&(d_rho[ijk_to_n(i+1, j+1, k, N_grid)]),   q*Xr*Yr*Zl);
        atomicAdd(&(d_rho[ijk_to_n(i+1, j, k+1, N_grid)]),   q*Xr*Yl*Zr);
        atomicAdd(&(d_rho[ijk_to_n(i, j+1, k+1, N_grid)]),   q*Xl*Yr*Zr);
        atomicAdd(&(d_rho[ijk_to_n(i+1, j+1, k+1, N_grid)]), q*Xr*Yr*Zr);
    }
}

void scatter_charge(Species *s, Grid *g)
{
    CUDA_ERROR(cudaDeviceSynchronize());
    scatter_charge_kernel<<<s->particleBlocks, s->particleThreads>>>(s->d_particles,
        s->q, g->d_rho, g->N_grid, g->dx, s->N_particles);
}


__device__ float gather_grid_to_particle(Particle *p, float *grid, int N_grid, float dx){
    float x = p->x;
    float y = p->y;
    float z = p->z;
    int i = position_to_grid_index(x, dx);
    int j = position_to_grid_index(y, dx);
    int k = position_to_grid_index(z, dx);

    float Xr = position_in_cell(x, dx)/dx;
    float Xl = 1 - Xr;
    float Yr = position_in_cell(y, dx)/dx;
    float Yl = 1 - Yr;
    float Zr = position_in_cell(z, dx)/dx;
    float Zl = 1 - Zr;

    float interpolated_scalar = 0.0f;
    interpolated_scalar += grid[ijk_to_n(i, j, k, N_grid)]      *Xl*Yl*Zl;
    interpolated_scalar += grid[ijk_to_n(i+1, j, k, N_grid)]    *Xr*Yl*Zl;
    interpolated_scalar += grid[ijk_to_n(i, j+1, k, N_grid)]    *Xl*Yr*Zl;
    interpolated_scalar += grid[ijk_to_n(i, j, k+1, N_grid)]    *Xl*Yl*Zr;
    interpolated_scalar += grid[ijk_to_n(i+1, j+1, k, N_grid)]  *Xr*Yr*Zl;
    interpolated_scalar += grid[ijk_to_n(i+1, j, k+1, N_grid)]  *Xr*Yl*Zr;
    interpolated_scalar += grid[ijk_to_n(i, j+1, k+1, N_grid)]  *Xl*Yr*Zr;
    interpolated_scalar += grid[ijk_to_n(i+1, j+1, k+1, N_grid)]*Xr*Yr*Zr;
    return interpolated_scalar;

}



__global__ void InitialVelocityStep_kernel(Particle *d_p, float q, float m, float *d_Ex,
        float *d_Ey, float *d_Ez, int N_particles, int N_grid, float dx, float dt){
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n<N_particles)
    {
        Particle *p = &(d_p[n]);
        //gather electric field
        float Ex = gather_grid_to_particle(p, d_Ex, N_grid, dx);
        float Ey = gather_grid_to_particle(p, d_Ey, N_grid, dx);
        float Ez = gather_grid_to_particle(p, d_Ez, N_grid, dx);

       //use electric field to accelerate particles
       p->vx -= 0.5f*dt*q/m*Ex;
       p->vy -= 0.5f*dt*q/m*Ey;
       p->vz -= 0.5f*dt*q/m*Ez;
    }
}
void InitialVelocityStep(Species *s, Grid *g, float dt)
{
    InitialVelocityStep_kernel<<<s->particleBlocks, s->particleThreads>>>(s->d_particles,
         s->q, s->m, g->d_Ex, g->d_Ey, g->d_Ez, s->N_particles, g->N_grid, g->dx, dt);
}

__global__ void ParticleKernel(Particle *d_p, float q, float m,
        float *d_Ex, float *d_Ey, float *d_Ez, int N_particles, int N_grid, float dx, float dt,
        float* d_block_v2s, float* d_block_Px, float* d_block_Py, float* d_block_Pz){
   __shared__ float v2_array[pThreads];
   __shared__ float Px_array[pThreads];
   __shared__ float Py_array[pThreads];
   __shared__ float Pz_array[pThreads];
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

       float old_vx = p->vx;
       float old_vy = p->vy;
       float old_vz = p->vz;

       //gather electric field
       float Ex = gather_grid_to_particle(p, d_Ex, N_grid, dx);
       float Ey = gather_grid_to_particle(p, d_Ey, N_grid, dx);
       float Ez = gather_grid_to_particle(p, d_Ez, N_grid, dx);

       //use electric field to accelerate particles
       p->vx += dt*q/m*Ex;
       p->vy += dt*q/m*Ey;
       p->vz += dt*q/m*Ez;


       v2_array[threadIdx.x] = old_vx * p->vx + old_vy * p->vy + old_vz * p->vz;
       Px_array[threadIdx.x] = old_vx * p->vx;
       Py_array[threadIdx.x] = old_vy * p->vy;
       Pz_array[threadIdx.x] = old_vz * p->vz;
       __syncthreads();


       for (int s = pThreads / 2; s > 0; s >>= 1)
       {
           if (threadIdx.x < s)
           {
               v2_array[threadIdx.x] += v2_array[threadIdx.x + s];
               Px_array[threadIdx.x] += Px_array[threadIdx.x + s];
               Py_array[threadIdx.x] += Py_array[threadIdx.x + s];
               Pz_array[threadIdx.x] += Pz_array[threadIdx.x + s];
           }
           __syncthreads();
       }

       if (threadIdx.x == 0)
       {
           d_block_v2s[blockIdx.x] = v2_array[0];
           d_block_Px[blockIdx.x] = Px_array[0];
           d_block_Py[blockIdx.x] = Py_array[0];
           d_block_Pz[blockIdx.x] = Pz_array[0];
       }

   }
}

__global__ void reduce_moments(float *d_arr, float *d_results, int N)
{
    __shared__ float sh_array[pThreads];
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    // sh_array[threadIdx.x] = 0;
    if (n < N){
        for (int s = blockDim.x / 2; s > 0; s >>= 1){
            if ( threadIdx.x < s)
            {
                sh_array[threadIdx.x] += d_arr[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x ==0){
            d_results[blockIdx.x] = sh_array[0];
            // printf("%d %f\n", blockIdx.x, d_results[blockIdx.x]);
        }
    }
}

void SpeciesPush(Species *s, Grid *g, float dt)
{
    ParticleKernel<<<s->particleBlocks, s->particleThreads>>>(s->d_particles,
        s->q, s->m, g->d_Ex, g->d_Ey, g->d_Ez, s->N_particles, g->N_grid, g->dx, dt,
        s->d_block_v2s, s->d_block_Px, s->d_block_Py, s->d_block_Pz);

    // printf("%d %d %d\n", (s->particleBlocks.x + pThreads - 1)/pThreads, s->particleThreads.x, (s->N_particles + pThreads - 1)/pThreads);
    CUDA_ERROR(cudaDeviceSynchronize());
    reduce_moments<<<(s->particleBlocks.x + pThreads - 1)/pThreads, s->particleThreads>>>(s->d_block_v2s, s->d_sum_v2s, (s->N_particles + pThreads - 1)/pThreads);
    reduce_moments<<<(s->particleBlocks.x + pThreads - 1)/pThreads, s->particleThreads>>>(s->d_block_Px, s->d_sum_Px, (s->N_particles + pThreads - 1)/pThreads);
    reduce_moments<<<(s->particleBlocks.x + pThreads - 1)/pThreads, s->particleThreads>>>(s->d_block_Py, s->d_sum_Py, (s->N_particles + pThreads - 1)/pThreads);
    reduce_moments<<<(s->particleBlocks.x + pThreads - 1)/pThreads, s->particleThreads>>>(s->d_block_Pz, s->d_sum_Pz, (s->N_particles + pThreads - 1)/pThreads);
    CUDA_ERROR(cudaDeviceSynchronize());
    // printf("%d %d %ld\n",1, (s->particleBlocks.x + pThreads - 1)/pThreads, (s->particleBlocks.x + pThreads - 1)/pThreads);
    reduce_moments<<<1, (s->particleBlocks.x + pThreads - 1)/pThreads>>>(s->d_sum_v2s, &(s->d_moments[0]), (s->particleBlocks.x + pThreads - 1)/pThreads);
    reduce_moments<<<1, (s->particleBlocks.x + pThreads - 1)/pThreads>>>(s->d_sum_Px, &(s->d_moments[1]), (s->particleBlocks.x + pThreads - 1)/pThreads);
    reduce_moments<<<1, (s->particleBlocks.x + pThreads - 1)/pThreads>>>(s->d_sum_Py, &(s->d_moments[2]), (s->particleBlocks.x + pThreads - 1)/pThreads);
    reduce_moments<<<1, (s->particleBlocks.x + pThreads - 1)/pThreads>>>(s->d_sum_Pz, &(s->d_moments[3]), (s->particleBlocks.x + pThreads - 1)/pThreads);

    CUDA_ERROR(cudaMemcpy(s->moments, s->d_moments, sizeof(float)*4, cudaMemcpyDeviceToHost));
    // printf("%f %f %f %f\n", s->moments[0], s->moments[1], s->moments[2], s->moments[3]);
    s->KE = s->moments[0] * 0.5f * s->m;
    s->Px = s->moments[1] * s->m;
    s->Py = s->moments[2] * s->m;
    s->Pz = s->moments[3] * s->m;
    // printf("%f %f %f %f\n", s->KE, s->Px, s->Py, s->Pz);
}

void dump_position_data(Species *s, char* name){
    // printf("Copying particles from GPU to device\n");
    CUDA_ERROR(cudaMemcpy(s->particles, s->d_particles, sizeof(Particle)*s->N_particles, cudaMemcpyDeviceToHost));
    // printf("Copied particles from GPU to device\n");
    FILE *initial_position_data = fopen(name, "w");
    for (int i =0; i<s->N_particles; i += 51)
    {
        Particle *p = &(s->particles[i]);
        fprintf(initial_position_data, "%f %f %f %f %f %f\n", p->x, p->y, p->z, p->vx, p->vy, p->vz);
    }
    // free(s->particles);
    fclose(initial_position_data);
}

void particle_cleanup(Species *s)
{
    CUDA_ERROR(cudaFree(s->d_particles));
    CUDA_ERROR(cudaFree(s->d_block_v2s));
    CUDA_ERROR(cudaFree(s->d_block_Px));
    CUDA_ERROR(cudaFree(s->d_block_Py));
    CUDA_ERROR(cudaFree(s->d_block_Pz));
}
