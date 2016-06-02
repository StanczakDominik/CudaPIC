#include "grid.h"

extern dim3 particleThreads;
extern dim3 particleBlocks;

__global__ void InitParticleArrays(Particle *d_p, float shiftx, float shifty, float shiftz);
struct Particle{
    //keeps information about the position of one particle in (6D) phase space (positions, velocities)
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

struct Species{
    //keeps information about one distinct group of particles
    float m; //mass
    float q; //charge

    //number of particles in group: not fully used yet
    long int N;

    Particle *particles;
    Particle *d_particles;

    Particle total_values;
    float total_v2;
    float total_vabs;
    float T;
    float kinetic_E;
    float potential_E;
};



__global__ void scatter_charge(Particle *d_P, float q, float* d_rho);
__device__ float gather_grid_to_particle(Particle *p, float *grid);
__global__ void InitialVelocityStep(Particle *d_p, float q, float m, float *d_Ex, float *d_Ey, float *d_Ez);
__global__ void ParticleKernel(Particle *d_p, float q, float m, float *d_Ex, float *d_Ey, float *d_Ez);
void init_species(Species *s, float shiftx, float shifty, float shiftz);
void dump_position_data(Species *s, char* name);
