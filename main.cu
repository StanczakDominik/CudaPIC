#include <stdio.h>
#include <iostream>
#include "grid.cuh"
#include "helpers.cuh"
#include "particles.cuh"
using namespace std;

#define SNAP_EVERY 10
#define NT 1000
#define dt 0.01f

#define N_particles_1_axis 128
#define N_particles  (N_particles_1_axis*N_particles_1_axis*N_particles_1_axis)


void init_timestep(Grid *g, Species *electrons,  Species *ions){
    reset_rho(g);
    scatter_charge(electrons, g);
    scatter_charge(ions, g);

    // debug_field_solver_uniform(g);
    field_solver(g);

    InitialVelocityStep(electrons, g, dt);
    InitialVelocityStep(ions, g, dt);
}


void timestep(Grid *g, Species *electrons,  Species *ions){
	//1. move particles, gather electric fields at their locations, accelerate particles
	SpeciesPush(electrons, g, dt);
	SpeciesPush(ions, g, dt);
	//potential TODO: sort particles?????
    //2. clear charge density for scattering fields to particles charge
    reset_rho(g);

    //3. gather charge from new particle position to grid
    scatter_charge(electrons, g);
    scatter_charge(ions, g);

    //4. use charge density to calculate field
    // debug_field_solver_uniform(g);
    field_solver(g);
}

int main(void){

    int N_grid = 32;

    cudaEvent_t startLoop, endLoop;
    cudaEventCreate(&startLoop);
    cudaEventCreate(&endLoop);


    Grid g;
    init_grid(&g, N_grid);
    CUDA_ERROR(cudaDeviceSynchronize());

    Species electrons;
    electrons.q = -ELECTRON_CHARGE;
    electrons.m = ELECTRON_MASS;
    electrons.N = N_particles;
    init_species(&electrons, g.dx*0.06f, g.dx*0.001f, g.dx*0.001f, 1, 0, 0, N_particles_1_axis, g.N_grid, g.dx);
    Species ions;
    // ions.q = ELECTRON_CHARGE;
    // ions.m = PROTON_MASS;
    ions.q = -ELECTRON_CHARGE;
    ions.m = ELECTRON_MASS;
    ions.N = N_particles;
    init_species(&ions, g.dx*0.05f, g.dx*0.001f, g.dx*0.001f, -1, 0, 0, N_particles_1_axis, g.N_grid, g.dx);

    char filename[50];
    sprintf(filename, "data/ions_positions_%d.dat", -1);
    dump_position_data(&ions, filename);
    sprintf(filename, "data/electrons_positions_%d.dat", -1);
    dump_position_data(&electrons, filename);

    init_timestep(&g, &electrons, &ions);

    printf("entering time loop\n");
    cudaEventSynchronize(startLoop);
    cudaEventRecord(startLoop);
    FILE *energy_data = fopen("energies.dat", "w");
    for(int i =0; i<=NT; i++){
        if (i % SNAP_EVERY == 0)
        {
            printf("Iteration %6d\n", i);
            sprintf(filename, "data/running_density_%d.dat", i);
            dump_density_data(&g, (char*)filename);
            sprintf(filename, "data/ions_positions_%d.dat", i);
            dump_position_data(&ions, filename);
            sprintf(filename, "data/electrons_positions_%d.dat", i);
            dump_position_data(&electrons, filename);
            fprintf(energy_data, "%d ", i);
            fprintf(energy_data, "%f %f ", g.rho_total, g.E_total);
            fprintf(energy_data, "%f %f ", electrons.KE, ions.KE);
            fprintf(energy_data, "%f %f ", electrons.Px, ions.Px);
            fprintf(energy_data, "%f %f ", electrons.Py, ions.Py);
            fprintf(energy_data, "%f %f ", electrons.Pz, ions.Pz);
            fprintf(energy_data, "%f\n", g.E_total + electrons.KE + ions.KE);
        }
        timestep(&g, &electrons, &ions);
    }
    fclose(energy_data);
    cudaDeviceSynchronize();
    cudaEventSynchronize(endLoop);
    cudaEventRecord(endLoop);
    printf("\nfinished time loop\n");
    float loopRuntimeMS = 0;
    cudaEventElapsedTime(&loopRuntimeMS, startLoop, endLoop);


    grid_cleanup(&g);
    particle_cleanup(&electrons);
    particle_cleanup(&ions);
}
