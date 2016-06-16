#include "grid.cuh"


__global__ void solve_poisson(float *d_kv, cufftComplex *d_F_rho,
        cufftComplex *d_F_Ex, cufftComplex *d_F_Ey, cufftComplex *d_F_Ez,
        int N_grid, int N_grid_all){
    /*solve poisson equation
    d_kv: wave vector
    d_F_rho: complex array of fourier transformed charge densities
    d_F_E(i):
    */
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int index = k*N_grid*N_grid + j*N_grid + i;
    if(i<N_grid && j<N_grid && k<N_grid){
        float k2inverse; //wave vector magnitude squared
        if (i==0 && j==0 && k ==0)    {
            k2inverse = 1.0f; //dodge a bullet with a division by zero
        }
        else
        {
            k2inverse = 1.0f/(d_kv[i]*d_kv[i] + d_kv[j]*d_kv[j] + d_kv[k]*d_kv[k]);
        }

        //see: Birdsall Langdon, Plasma Physics via Computer Simulation, page 19
        d_F_Ex[index].x = -d_kv[i]*d_F_rho[index].x*k2inverse/EPSILON_ZERO;
        d_F_Ex[index].y = -d_kv[i]*d_F_rho[index].y*k2inverse/EPSILON_ZERO;

        d_F_Ey[index].x = -d_kv[j]*d_F_rho[index].x*k2inverse/EPSILON_ZERO;
        d_F_Ey[index].y = -d_kv[j]*d_F_rho[index].y*k2inverse/EPSILON_ZERO;

        d_F_Ez[index].x = -d_kv[k]*d_F_rho[index].x*k2inverse/EPSILON_ZERO;
        d_F_Ez[index].y = -d_kv[k]*d_F_rho[index].y*k2inverse/EPSILON_ZERO;


    }
}

__global__ void real2complex(float *input, cufftComplex *output, int N_grid){
    //converts array of floats to array of real complex numbers
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int index = k*N_grid*N_grid + j*N_grid + i;

    if(i<N_grid && j<N_grid && k<N_grid)    {
        output[index].x = input[index];
        output[index].y = 0.0f;
    }
}
__global__ void complex2real(cufftComplex *input, float *output, int N_grid, int N_grid_all){
    //converts array of complex inputs to floats (discards)
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int index = k*N_grid*N_grid + j*N_grid + i;

    if(i<N_grid && j<N_grid && k<N_grid){
        output[index] = input[index].x/float(N_grid_all);
    }
}

__global__ void scale_down_after_fft(float *d_Ex, float *d_Ey, float *d_Ez, int N_grid, int N_grid_all){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int index = k*N_grid*N_grid + j*N_grid + i;

    if(i<N_grid && j<N_grid && k<N_grid){
        d_Ex[index] /= float(N_grid_all);
        d_Ey[index] /= float(N_grid_all);
        d_Ez[index] /= float(N_grid_all);
    }
}

__global__ void set_grid_array_to_value(float *arr, float value, int N_grid){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int index = k*N_grid*N_grid + j*N_grid + i;

    if((i<N_grid) && (j<N_grid) && (k<N_grid)){
        arr[index] = value;
    }
}

void reset_rho(Grid *g)
{
    CUDA_ERROR(cudaDeviceSynchronize());
    set_grid_array_to_value<<<g->gridBlocks, g->gridThreads>>>(g->d_rho, 0, g->N_grid);
}


/*
* HIGH LEVEL KERNEL WRAPPERS
*/

void init_grid(Grid *g, int N_grid){
    int N_grid_all = N_grid * N_grid * N_grid;
    g->N_grid_all = N_grid_all;
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
    g->dx = (L/float(N_grid));
    g->dy = g->dx;
    g->dz = g->dx;

    g->gridThreads = dim3(N_grid/2,N_grid/2,N_grid/2);
    g->gridBlocks = dim3((N_grid+g->gridThreads.x-1)/g->gridThreads.x,
        (N_grid + g->gridThreads.y - 1)/g->gridThreads.y, (N_grid+g->gridThreads.z-1)/g->gridThreads.z);

    // printf("%d %d %d\n", g->gridThreads.x, g->gridThreads.y, g->gridThreads.z);
    // printf("%d %d %d\n", g->gridBlocks.x, g->gridBlocks.y, g->gridBlocks.z);
    CUDA_ERROR(cudaMalloc((void**)&(g->d_kv), sizeof(float)*N_grid));
    CUDA_ERROR(cudaMemcpy(g->d_kv, g->kv, sizeof(float)*N_grid, cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMalloc((void**)&(g->d_F_rho), sizeof(cufftComplex)*N_grid_all));
    CUDA_ERROR(cudaMalloc((void**)&(g->d_F_Ex), sizeof(cufftComplex)*N_grid_all));
    CUDA_ERROR(cudaMalloc((void**)&(g->d_F_Ey), sizeof(cufftComplex)*N_grid_all));
    CUDA_ERROR(cudaMalloc((void**)&(g->d_F_Ez), sizeof(cufftComplex)*N_grid_all));


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

void field_solver(Grid *g){
    CUDA_ERROR(cudaDeviceSynchronize());
    cufftExecR2C(g->plan_forward, g->d_rho, g->d_F_rho);
    CUDA_ERROR(cudaDeviceSynchronize());
    solve_poisson<<<g->gridBlocks, g->gridThreads>>>(g->d_kv, g->d_F_rho, g->d_F_Ex, g->d_F_Ey, g->d_F_Ez, g->N_grid, g->N_grid_all);
        //gridBlocks, gridThreads allocated
        //g->d_kv calculated looks like properly
        //d->F_rho
    CUDA_ERROR(cudaDeviceSynchronize());
    cufftExecC2R(g->plan_backward, g->d_F_Ex, g->d_Ex);
    cufftExecC2R(g->plan_backward, g->d_F_Ey, g->d_Ey);
    cufftExecC2R(g->plan_backward, g->d_F_Ez, g->d_Ez);
    CUDA_ERROR(cudaDeviceSynchronize());

    scale_down_after_fft<<<g->gridBlocks, g->gridThreads>>>(g->d_Ex, g->d_Ey, g->d_Ez, g->N_grid, g->N_grid_all);
    CUDA_ERROR(cudaDeviceSynchronize());
}


void dump_density_data(Grid *g, char* name){
    CUDA_ERROR(cudaMemcpy(g->rho, g->d_rho, sizeof(float)*(g->N_grid_all), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(g->Ex, g->d_Ex, sizeof(float)*(g->N_grid_all), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(g->Ey, g->d_Ey, sizeof(float)*(g->N_grid_all), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(g->Ez, g->d_Ez, sizeof(float)*(g->N_grid_all), cudaMemcpyDeviceToHost));
    FILE *density_data = fopen(name, "w");
    float rho_total = 0.0f;
    for (int n = 0; n < g->N_grid_all; n++)
    {
        fprintf(density_data, "%f %.2f %.2f %.2f\n", g->rho[n], g->Ex[n], g->Ey[n], g->Ez[n]);
        rho_total += g->rho[n];
    }
    printf("rho total: %f\n", rho_total);
}



/*
*   DEBUG SOLVERS
*
*/
void debug_field_solver_uniform(Grid *g){
    float* linear_field_x = new float[g->N_grid_all];
    float* linear_field_y = new float[g->N_grid_all];
    float* linear_field_z = new float[g->N_grid_all];
    for(int i = 0; i<g->N_grid;  i++){
        for(int j = 0; j<g->N_grid;  j++){
            for(int k = 0; k<g->N_grid;  k++){
                int index = i*g->N_grid*g->N_grid + j*g->N_grid + k;
                linear_field_x[index] = 1000;
                linear_field_y[index] = 0;
                linear_field_z[index] = 0;
            }
        }
    }
    cudaMemcpy(g->d_Ex, linear_field_x, sizeof(float)*g->N_grid_all, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_Ey, linear_field_y, sizeof(float)*g->N_grid_all, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_Ez, linear_field_z, sizeof(float)*g->N_grid_all, cudaMemcpyHostToDevice);
}
void debug_field_solver_sine(Grid *g)
{
    float* linear_field_x = new float[g->N_grid_all];
    float* linear_field_y = new float[g->N_grid_all];
    float* linear_field_z = new float[g->N_grid_all];
    for(int i = 0; i<g->N_grid;  i++){
        for(int j = 0; j<g->N_grid;  j++){
            for(int k = 0; k<g->N_grid;  k++){
                int index = i*g->N_grid*g->N_grid + j*g->N_grid + k;
                linear_field_x[index] = 1000*sin(2*M_PI*((float)k/(float)g->N_grid));
                linear_field_y[index] = 1000*sin(2*M_PI*((float)j/(float)g->N_grid));
                linear_field_z[index] = 1000*sin(2*M_PI*((float)i/(float)g->N_grid));
            }
        }
    }
    cudaMemcpy(g->d_Ex, linear_field_x, sizeof(float)*g->N_grid_all, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_Ey, linear_field_y, sizeof(float)*g->N_grid_all, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_Ez, linear_field_z, sizeof(float)*g->N_grid_all, cudaMemcpyHostToDevice);
}
