const int N_particles_1_axis = 16; //should be maybe connected to threadsPerBlock somehow
const int N_particles = N_particles_1_axis*N_particles_1_axis*N_particles_1_axis; //does this compile with const? //2^4^3 = 2^7 = 128 
const float L = 1.;
const int N_grid_1_axis = 8;
const int N_gird = N_grid_1_axis*N_grid_1_axis*N_grid_1_axis;
// const float Ly = 1.;
// const float Lz = 1.;

size_t particle_array_size = N_particles*sizeof(float);
size_t grid_array_size = N_grid*sizeof(float);

/*
Assumptions:
q=1
m=1
L = 1
*/

dim3 particleThreads(16,3);
dim3 particleBlocks(N_particles/threadsPerBlock.x);

dim3 gridThreads(16,16);
dim3 gridBlocks(N_grid/gridThreads.x, N_grid/gridThreads.y, N_grid/grid_threads.z);

__global__ void InitParticleArrays(float* R, float* V, int N_particles, int N_particles_1_axis, float L_axis)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = threadIdx.y;
    if (i<N_particles)
    {
        R[i][j] = L_axis/float(N_particles_1_axis) * i;
        V[i][j] = 0.;
    }
}

__global__ void InitGridArrays(float* ChargeDensity, float* Ex, float* Ey, float* Ez, int N_grid)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = threadIdx.z * blockIdx.z + threadIdx.z;
    if (i<N_grid and j < N_grid and k < N_grid)
    {
        ChargeDensity[i][j][k] = 0.;
        Ex[i][j][k] = 0.;
        Ey[i][j][k] = 0.;
        Ez[i][j][k] = 0.;
    }
}

    
__global__ void InitialVelocityStep(float* V, float* E, float dt, int N_particles)
{
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = threadIx.y;
    if(i<N_particles)
    {
        V[i][j] -= 0.5*dt*E[i][j]; // TODO: times q/m eventually?
    }
}
///todo: can you make a __device__ function in cuda that actually returns numbers? can that access indices etc? I could replace E[i][j] with a function returning -2(R-L/2) for now
__device__ float InterpolateElectricField(int j, float x, float y, float z)//, float* E)
{
    if (j==0)
    {
        return -2*(x-0.5);
    }
    else if (j==1)
    {
        return -2*(y-0.5);        
    }
    else (j==2)
    {
        return -2*(z-0.5);
    }
}
__global__ void ParticleKernel(float* R, float* V, float* Ex, float* Ey, float*Ez, float dt, int N_particles, float L) //eventually: grid position array (and THAT can be 3d!!!)!, grid charge array, grid field arrays
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int j = threadIx.y;
   if(i<N_particles)
   {
       //particle kernel: update positions, interpolate fields, update velocities, deposit charges
        R[i][j] += V[i][j]*dt;
        R[i][j] %= L; //can you do that?
        V[i][j] += InterpolateElectricField(int j, R[i][0], R[i][1], R[i][2]);//, Ex, Ey, Ez);
        //todo: charge deposition
        //I[i][j] = int(R[i][j] / dj);
        //I goes on out and is returned for the grid thing    
   }        
}
//charge deposition has to happen somewhere here, maybe via... histogram? scan? gotta figure something out.
__global__ void GridKernel(float *GridR, float* GridChargeDensity, float* Ex, float* Ey, float* Ez, int N_grid)
{
    
}

int main(void)
{
    //TODO: routine checks for cuda status


    //allocate space for particles
    
    float* d_R;
    float* d_V;
    float* d_CD;
//     float* d_GridX;     //czy aby one na pewno są konieczne do alokowania? ani się to nie zmienia... tak naprawdę mając i*dx...
//     float* d_GridY;
//     float* d_GridZ;
    float* d_Ex;
    float* d_Ey;
    float* d_Ez;
//     int* d_I;
    cudaMalloc(&d_R, particle_array_size);
    cudaMalloc(&d_V, particle_array_size);
    cudaMalloc(&d_Ex, grid_array_size);
    cudaMalloc(&d_Ey, grid_array_size);
    cudaMalloc(&d_Ez, grid_array_size);
    cudaMalloc(&d_CD, grid_array_size);
//     cudaMalloc(&d_I, particle_array_size); //indices on grid
    
    InitParticleArrays<<particleBlocks, particleThreads>>(d_R, d_V, N_particles, N_particles_1_axis, L);
    InitGridArrays<<gridBlock, gridThreads>>(d_CD, d_Ex, d_Ey, d_Ez);
    
    for(int i =0; i<NT; i++i)
    {
        ParticleKernel<<particleBlocks, particleThreads>>();
        GridKernel<<gridBlocks, gridThreads>>();
    }
    

    /*
    kernel1: set particle positions to uniform spatial distribution, zero 
    problem1: given int N particles, how to set their x, y, z
    1d: easy, given Nx particles, Nx/Lx * idX
    2d: sorta easy because [(Nx/Lx) * idX, (Ny/Ly) * idY]
    how do we get Nx, Ny?
    In 2d we would have N particles, that's sqrt(N) particles for each row
    problem: sqrt not necessary a float
    turn it around
    */

}