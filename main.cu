#include <stdio.h>
#include <iostream>
#include "grid.cuh"
#include "helpers.cuh"
#include "particles.cuh"
using namespace std;

#define SNAP_EVERY 50
#define NT 10000
#define dt 0.01f
#define N_grid 16
#define N_grid_all (N_grid *N_grid * N_grid)
dim3 gridThreads(N_grid/2,N_grid/2,N_grid/2);
dim3 gridBlocks((N_grid+gridThreads.x-1)/gridThreads.x, (N_grid + gridThreads.y - 1)/gridThreads.y, (N_grid+gridThreads.z-1)/gridThreads.z);
#define dx (L/float(N_grid))
#define dy dx
#define dz dx


void init_timestep(Grid *g, Species *electrons,  Species *ions){
    set_grid_array_to_value<<<gridBlocks, gridThreads>>>(g->d_rho, 0);
    CUDA_ERROR(cudaDeviceSynchronize());
    scatter_charge<<<particleBlocks, particleThreads>>>(electrons->d_particles, electrons->q, g->d_rho);
    CUDA_ERROR(cudaDeviceSynchronize());
    scatter_charge<<<particleBlocks, particleThreads>>>(ions->d_particles, ions->q, g->d_rho);
    CUDA_ERROR(cudaDeviceSynchronize());

    // debug_field_solver_uniform(g);
    field_solver(g);
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
    CUDA_ERROR(cudaDeviceSynchronize());
    scatter_charge<<<particleBlocks, particleThreads>>>(ions->d_particles, ions->q, g->d_rho);
    CUDA_ERROR(cudaDeviceSynchronize());

    //4. use charge density to calculate field
    field_solver(g);
    // debug_field_solver_uniform(g);
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaDeviceSynchronize());
}

int main(void){


    cudaEvent_t startLoop, endLoop;
    cudaEventCreate(&startLoop);
    cudaEventCreate(&endLoop);


    Grid g;
    init_grid(&g);

    Species electrons;
    electrons.q = -ELECTRON_CHARGE;
    electrons.m = ELECTRON_MASS;
    electrons.N = N_particles;
    init_species(&electrons, dx*0.1f, dy*0.1f, dz*0.1f, 0, 0, 0);

    Species ions;
    ions.q = ELECTRON_CHARGE;
    ions.m = PROTON_MASS;
    ions.N = N_particles;
    init_species(&ions, dx*0.05f, dx*0.05f, dx*0.05f, 0, 0, 0);

    CUDA_ERROR(cudaGetLastError());
    dump_position_data(&ions, "data/ions_positions.dat");
    dump_position_data(&electrons, "data/electrons_positions.dat");
    dump_density_data(&g, (char*)"data/initial_density.dat");

    init_timestep(&g, &electrons, &ions);

    printf("entering time loop\n");
    cudaEventSynchronize(startLoop);
    cudaEventRecord(startLoop);
    char filename[50];
    for(int i =0; i<NT; i++){
        if (i % SNAP_EVERY == 0)
        {
            sprintf(filename, "data/running_density_%d.dat", i);
            dump_running_density_data(&g, (char*)filename);
            sprintf(filename, "data/ions_positions_%d.dat", i);
            dump_position_data(&ions, filename);
            sprintf(filename, "data/electrons_positions_%d.dat", i);
            dump_position_data(&electrons, filename);
        }
        timestep(&g, &electrons, &ions);
        printf("Iteration %d\r", i);
    }

    cudaDeviceSynchronize();
    cudaEventSynchronize(endLoop);
    cudaEventRecord(endLoop);
    printf("\nfinished time loop\n");
    float loopRuntimeMS = 0;
    cudaEventElapsedTime(&loopRuntimeMS, startLoop, endLoop);


    sprintf(filename, "data/running_density_%d.dat", NT);
    dump_running_density_data(&g, (char*)filename);
    sprintf(filename, "data/ions_positions_%d.dat", NT);
    dump_position_data(&ions, filename);
    sprintf(filename, "data/electrons_positions_%d.dat", NT);
    dump_position_data(&electrons, filename);

    printf("Particles Threads per block Blocks Runtime\n");
    printf("%8d %17d %6d %f\n", N_particles, particleThreads.x, particleBlocks.x, loopRuntimeMS);
    if (loopRuntimeMS > 0.0001)
    {
        char* filename = new char[100];
        sprintf(filename, "data/pb_%d_%d_%d.bdat", N_particles, particleThreads.x, particleBlocks.x);
        FILE *benchmark = fopen(filename, "w");
        fprintf(benchmark, "Particles Threads per block Blocks\tRuntime [ms]\n");
        fprintf(benchmark, "%8d %17d %6d %f\n", N_particles, particleThreads.x, particleBlocks.x, loopRuntimeMS);
        fclose(benchmark);
    }
    else
    {
        printf("Not saved!\n");
    }

    dump_density_data(&g, (char*)"data/final_density.dat");


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
