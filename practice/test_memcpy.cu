#include <stdio.h>
#include <iostream>
using namespace std;
#define N_particles 512
#define L 1.0f
#define N_particles_1_axis 8

dim3 particleThreads(64);
dim3 particleBlocks(N_particles/particleThreads.x);

static void CUDA_ERROR( cudaError_t err)
{
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s, exiting\n", cudaGetErrorString(err));
        exit(-1);
    }
}

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

__global__ void InitParticleArrays(Particle *s){
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n<N_particles)
    {
        Particle *p = &(s[n]);
        (*p).x = L/float(N_particles_1_axis)*(n%N_particles_1_axis);
        (*p).y = L/float(N_particles_1_axis)*(n/N_particles_1_axis)/float(N_particles_1_axis);
        (*p).z = L/float(N_particles_1_axis)*(n/N_particles_1_axis/N_particles_1_axis);
        (*p).vx = 0.0f;
        (*p).vy = 0.0f;
        (*p).vz = 0.0f;
    }
}

void init_species(Species *s){
    s->particles = new Particle[N_particles];
    CUDA_ERROR(cudaMalloc((void**)&(s->d_particles), sizeof(Particle)*N_particles));
    cout << "initializing particles" << endl;
    InitParticleArrays<<<particleBlocks, particleThreads>>>(s->d_particles);
}

void dump_position_data(Species *s, char* name){
    cout << "Copying particles from GPU to device"<< endl;
    CUDA_ERROR(cudaMemcpy(s->particles, s->d_particles, sizeof(Particle)*N_particles, cudaMemcpyDeviceToHost));
    cout << "Copied particles from GPU to device"<< endl;
    FILE *initial_position_data = fopen(name, "w");
    for (int i =0; i<N_particles; i++)
    {
        Particle *p = &(s->particles[i]);
        fprintf(initial_position_data, "%f %f %f\n", p->x, p->y, p->z);
    }
}

int main(){
    Species electrons;
    electrons.q = -1.0f;
    electrons.m = 1.0f;
    electrons.N = N_particles;
    init_species(&electrons);

    // electrons.particles = new Particle[N_particles];
    // CUDA_ERROR(cudaMalloc((void**)&electrons.d_particles, sizeof(Particle)*N_particles));
    // cout << "initializing particles" << endl;
    // InitParticleArrays<<<particleBlocks, particleThreads>>>(electrons.d_particles);

    cout << electrons.q << endl << electrons.m << endl << electrons.N << endl << electrons.particles[1].x << endl;

    cout << "dumping positions" << endl;
    dump_position_data(&electrons, "init_position.dat");

    // cout << "Copying particles from GPU to device"<< endl;
    // CUDA_ERROR(cudaMemcpy(electrons.particles, electrons.d_particles, sizeof(Particle)*N_particles, cudaMemcpyDeviceToHost));
    // cout << "Copied particles from GPU to device"<< endl;
    // FILE *initial_position_data = fopen("blah.dat", "w");
    // for (int i =0; i<N_particles; i++)
    // {
    //     Particle *p = &(electrons.particles[i]);
    //     fprintf(initial_position_data, "%f %f %f\n", p->x, p->y, p->z);
    // }
}
