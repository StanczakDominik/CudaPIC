#include "particles.cuh"

dim3 particleThreads(N_particles_1_axis);
dim3 particleBlocks((N_particles+particleThreads.x - 1)/particleThreads.x);


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
    //TODO: redo this using a reduce
    atomicAdd(&(d_rho[N_grid * N_grid * ((k)%N_grid) + N_grid * ((j)%N_grid) + ((i)%N_grid)]), q*Xl*Yl*Zl);
    atomicAdd(&(d_rho[N_grid * N_grid * ((k)%N_grid) + N_grid * ((j)%N_grid) + ((i+1)%N_grid)]), q*Xr*Yl*Zl);
    atomicAdd(&(d_rho[N_grid * N_grid * ((k)%N_grid) + N_grid * ((j+1)%N_grid) + ((i)%N_grid)]), q*Xl*Yr*Zl);
    atomicAdd(&(d_rho[N_grid * N_grid * ((k+1)%N_grid) + N_grid * ((j)%N_grid) + ((i)%N_grid)]), q*Xl*Yl*Zr);
    atomicAdd(&(d_rho[N_grid * N_grid * ((k)%N_grid) + N_grid * ((j+1)%N_grid) + ((i+1)%N_grid)]), q*Xr*Yr*Zl);
    atomicAdd(&(d_rho[N_grid * N_grid * ((k+1)%N_grid) + N_grid * ((j)%N_grid) + ((i+1)%N_grid)]), q*Xr*Yl*Zr);
    atomicAdd(&(d_rho[N_grid * N_grid * ((k+1)%N_grid) + N_grid * ((j+1)%N_grid) + ((i)%N_grid)]), q*Xl*Yr*Zr);
    atomicAdd(&(d_rho[N_grid * N_grid * ((k+1)%N_grid) + N_grid * ((j+1)%N_grid) + ((i+1)%N_grid)]), q*Xr*Yr*Zr);
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
    //TODO: zafunkcjować ten kawałek
    interpolated_scalar += grid[N_grid * N_grid * ((k)%N_grid) + N_grid * ((j)%N_grid) + ((i)%N_grid)]*Xl*Yl*Zl;
    interpolated_scalar += grid[N_grid * N_grid * ((k)%N_grid) + N_grid * ((j)%N_grid) + ((i+1)%N_grid)]*Xr*Yl*Zl;
    interpolated_scalar += grid[N_grid * N_grid * ((k)%N_grid) + N_grid * ((j+1)%N_grid) + ((i)%N_grid)]*Xl*Yr*Zl;
    interpolated_scalar += grid[N_grid * N_grid * ((k+1)%N_grid) + N_grid * ((j)%N_grid) + ((i)%N_grid)]*Xl*Yl*Zr;
    interpolated_scalar += grid[N_grid * N_grid * ((k)%N_grid) + N_grid * ((j+1)%N_grid) + ((i+1)%N_grid)]*Xr*Yr*Zl;
    interpolated_scalar += grid[N_grid * N_grid * ((k+1)%N_grid) + N_grid * ((j)%N_grid) + ((i+1)%N_grid)]*Xr*Yl*Zr;
    interpolated_scalar += grid[N_grid * N_grid * ((k+1)%N_grid) + N_grid * ((j+1)%N_grid) + ((i)%N_grid)]*Xl*Yr*Zr;
    interpolated_scalar += grid[N_grid * N_grid * ((k+1)%N_grid) + N_grid * ((j+1)%N_grid) + ((i+1)%N_grid)]*Xr*Yr*Zr;
    return interpolated_scalar;

}



__global__ void InitialVelocityStep(Particle *d_p, float q, float m, float *d_Ex, float *d_Ey, float *d_Ez){
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n<N_particles)
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
    printf("initializing particles\n");
    InitParticleArrays<<<particleBlocks, particleThreads>>>(s->d_particles, shiftx, shifty, shiftz);
}

void dump_position_data(Species *s, char* name){
    // printf("Copying particles from GPU to device\n");
    CUDA_ERROR(cudaMemcpy(s->particles, s->d_particles, sizeof(Particle)*N_particles, cudaMemcpyDeviceToHost));
    // printf("Copied particles from GPU to device\n");
    FILE *initial_position_data = fopen(name, "w");
    for (int i =0; i<N_particles; i += 51)
    {
        Particle *p = &(s->particles[i]);
        fprintf(initial_position_data, "%f %f %f %f %f %f\n", p->x, p->y, p->z, p->vx, p->vy, p->vz);
    }
    // free(s->particles);
    fclose(initial_position_data);
}

// __global__ void diagnostic_reduction_kernel(Species s)
// {
//     int n = blockDim.x * blockIdx.x + threadIdx.x;
//     if(n<N_particles)
//     {
//         Particle *p = &(s.d_particles[n]);
//         float rx = p->x;
//         float ry = p->y;
//         float rz = p->z;
//         float vx = p->vx;
//         float vy = p->vy;
//         float vz = p->vz;
//         float v2 = vx*vx + vy*vy + vz*vz;
//         float vabs = sqrt(v2);
//         //TODO:
//         //  particle field energy requires rewrite
//         //  to keep interpolated field as variable in particle
//         //  rel. easy
//
//         //reduce above variables
//     }
//     if(n == 0)
//     {
//         //s.total_values = reduced variables
//     }
// }
//
// void diagnostics(Species *s)
// {
//     /*
//     calculates:
//     mean velocity
//     mean square of velocity
//     variance as mean square of velocity - mean velocity squared
//
//     kinetic energy
//     potential energy of particles
//     field energy
//     for p in particles:
//
//     warte uśrednienia są:
//     vx, vy, vz, |v|, v^2
//
//     TODO: uśrednić na siatkę!!!!!! wtedy widać jak to ewoluuje!
//
//     1. loop po cząstkach
//         * vx
//         * vy
//         * vz
//         * v^2 = vx^2 + vy^2 + vz^2
//         * |v| = sqrt(v^2)
//         * V(r) (później)
//     2. reduce all powyższe (mozna inplace)
//     3. analiza danych:
//         * podzielić przez N_particles, średnie wielkości
//         * energia kinetyczna: 0.5 m sum v^2
//         * energia potencjalna: 0.5 q sum V(r)
//         * temperatura: 0.5 m (<v^2> - <v>^2)
//     */
//
//     diagnostic_reduction_kernel<<<particleBlocks, particleThreads>>>(s);
//     // float total_kinetic_energy = 0.5f * s.m * s.total_v2;
//     // float avg_modV = s.total_vabs / s.N;
//     // float avg_v2 = s.total_v2 / s.N;
//     // float temperature = 0.5f * s.m * (avg_v2 - avg_modV * avg_modV);
// }
