#include <cufft.h>
#include <iostream>
#include <fstream>
#define BSZ 32
using namespace std;
__global__ void solve_poisson(cufftComplex *ft, cufftComplex *ft_k, float*k, int N){
    int i = threadIdx.x + blockIdx.x*BSZ;
    int j = threadIdx.y + blockIdx.y*BSZ;
    int index = j*N+i;

    if (i<N && j<N)
    {
        float k2 = k[i] * k[i] + k[j]*k[j];
        if (i==0 && j == 0) {k2 = 1.0f;}                                 //a to jest takie w chuj ciekawe dlaczego oni to tak robią wtf
        ft_k[index].x = -ft[index].x/k2;
        ft_k[index].y = -ft[index].y/k2;
    }
}

__global__ void real2complex(float *f, cufftComplex *fc, int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = j*N + i;
    if(i<N && j < N)
    {
        fc[index].x = f[index];                 //ustawia rzeczywista
        fc[index].y = 0.0f;                     //zeruje urojona
    }
}

__global__ void complex2real(cufftComplex *fc, float *f, int N){
    int i = threadIdx.x + blockIdx.x * BSZ;
    int j = threadIdx.y + blockIdx.y * BSZ;
    int index = j*N+i;

    if (i<N && j < N)
    {
        f[index] = fc[index].x/((float)N*(float)N);        //divide by number of elements to recover value
    }
}

void write_to_file(char* filename, float* f, int N){
    ofstream file;
    file.open(filename);
    for(int i = 0; i<N; i++)
        for (int j = 0; j<N; j++)
        {
            file << f[N*j+i];
            file << "\n";
        }
    file.close();
}

int main(){
    int N = 64;
    float xmax = 1.0f, xmin = 0.0f, ymin = 0.0f,
          h = (xmax-xmin)/((float)N), s = 0.1, s2 = s*s;
    float *x = new float[N*N], *y = new float[N*N], *u = new float[N*N],
          *f = new float[N*N], *u_a = new float[N*N], *err = new float[N*N];

    float r2;

    for (int j = 0; j<N; j++)
        for (int i=0; i<N; i++)
        {
            x[N*j + i] = xmin + i*h; //allocate position matrix
            y[N*j + i] = ymin + j*h;

            r2 = (x[N*j+i]-0.5)*(x[N*j+i]-0.5) + (y[N*j+i]-0.5)*(y[N*j+i]-0.5);
            f[N*j+i] = (r2-2*s2)/(s2*s2)*exp(-r2/(2*s2));
            u_a[N*j+i] = exp(-r2/(2*s2));
        }

    float *k = new float[N];
    for (int i = 0; i<=N/2; i++)
    {
        k[i] = i*2*M_PI;
    }
    for(int i = N/2+1; i<N; i++)
    {
        k[i] = (i-N)*2*M_PI;
    }


    //the cuda begins here
    //allocate arrays upon yon device
    float *k_d, *f_d, *u_d;
    cudaMalloc((void**)&k_d, sizeof(float)*N);
    cudaMalloc((void**)&f_d, sizeof(float)*N*N);
    cudaMalloc((void**)&u_d, sizeof(float)*N*N);

    cudaMemcpy(k_d, k, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(f_d, f, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    cufftComplex *ft_d, *f_dc, *ft_d_k, *u_dc;
    cudaMalloc ((void**)&ft_d, sizeof(cufftComplex)*N*N);
    cudaMalloc ((void**)&ft_d_k, sizeof(cufftComplex)*N*N);
    cudaMalloc ((void**)&f_dc, sizeof(cufftComplex)*N*N);
    cudaMalloc ((void**)&u_dc, sizeof(cufftComplex)*N*N);

    cout << "N-0.5)/BSZ: " << int((N-0.5)/BSZ) + 1 << endl;
    dim3 dimGrid(int((N-0.5)/BSZ) + 1, int((N-0.5)/BSZ) + 1);
    dim3 dimBlock(BSZ, BSZ);

    real2complex<<<dimGrid, dimBlock>>>(f_d, f_dc, N);

    cufftHandle plan;
    cufftPlan2d(&plan, N, N, CUFFT_C2C);
    cufftExecC2C(plan, f_dc, ft_d, CUFFT_FORWARD);

    solve_poisson<<<dimGrid, dimBlock>>>(ft_d, ft_d_k, k_d, N);
    cufftExecC2C(plan, ft_d_k, u_dc, CUFFT_INVERSE);
    complex2real<<<dimGrid, dimBlock>>>(u_dc, u_d, N);
    cudaMemcpy(u, u_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);


    cufftDestroy(plan);
    cudaFree(k_d);
    cudaFree(f_d);
    cudaFree(u_d);

    float constant = u[0];
    for (int i =0; i<N*N; i++)
    {
        // cout << u[i] << endl;
        u[i] -= constant; //subtract u[0] to force arbitrary constant to be 0
    }

    write_to_file("x.dat", x, N);
    write_to_file("y.dat", y, N);
    write_to_file("u.dat", u, N);
    write_to_file("u_a.dat", u_a, N);
    write_to_file("f.dat", f, N);
}
