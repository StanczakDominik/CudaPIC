#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <iostream>
#include "grid.cuh"
#include "helpers.cuh"
#include "particles.cuh"
using namespace std;

void init_timestep(Grid *g, Species *electrons,  Species *ions){
    set_grid_array_to_value<<<gridBlocks, gridThreads>>>(g->d_rho, 0);
    CUDA_ERROR(cudaDeviceSynchronize());
    scatter_charge<<<particleBlocks, particleThreads>>>(electrons->d_particles, electrons->q, g->d_rho);
    CUDA_ERROR(cudaDeviceSynchronize());
    scatter_charge<<<particleBlocks, particleThreads>>>(ions->d_particles, ions->q, g->d_rho);
    CUDA_ERROR(cudaDeviceSynchronize());

    // debug_field_solver_sine(g);
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
    init_species(&electrons, L/100.0f, 0, 0);

    Species ions;
    ions.q = +ELECTRON_CHARGE;
    ions.m = PROTON_MASS;
    ions.N = N_particles;
    init_species(&ions, 0, 0, 0);
    //TODO: initialize for two stream instability
    init_timestep(&g, &electrons, &ions);

    CUDA_ERROR(cudaGetLastError());
    // dump_position_data(&ions, "ions_positions.dat");
    // dump_position_data(&electrons, "electrons_positions.dat");
    dump_density_data(&g, "initial_density.dat");

    cout << "entering time loop" << endl;
    cudaEventSynchronize(startLoop);
    cudaEventRecord(startLoop);
    for(int i =0; i<NT; i++){
        char* filename = new char[100];
        sprintf(filename, "gfx/running_density_%d.dat", i);
        dump_running_density_data(&g, filename);
        timestep(&g, &electrons, &ions);
        printf("Iteration %d\r", i);
    }

    cudaDeviceSynchronize();
    cudaEventSynchronize(endLoop);
    cudaEventRecord(endLoop);
    cout << endl << "finished time loop" << endl;

    float loopRuntimeMS = 0;
    cudaEventElapsedTime(&loopRuntimeMS, startLoop, endLoop);

    printf("Particles Threads per block Blocks Runtime\n");
    printf("%8d %17d %6d %f\n", N_particles, particleThreads.x, particleBlocks.x, loopRuntimeMS);
    if (loopRuntimeMS > 0.0001)
    {
        char* filename = new char[100];
        sprintf(filename, "benchmark/pb_%d_%d_%d.bdat", N_particles, particleThreads.x, particleBlocks.x);
        FILE *benchmark = fopen(filename, "w");
        fprintf(benchmark, "Particles Threads per block Blocks\tRuntime\n");
        fprintf(benchmark, "%8d %17d %6d %f\n", N_particles, particleThreads.x, particleBlocks.x, loopRuntimeMS);
        fclose(benchmark);
    }
    else
    {
        printf("Not saved!\n");
    }

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
