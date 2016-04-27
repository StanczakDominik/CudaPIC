# CudaPIC
### Because the good puns as names are all taken

A CUDA Particle-in-cell plasma simulation code
* Periodic boundary conditions for particles
* Particles are allocated on the GPU only
* Leapfrog particle pusher
* Fields are calculated on a discrete uniform grid  using a Fast Fourier Transform method
* Charge is gathered from particle positions in a terribly inefficient way using AtomicAdd
(this may or may be subject to change later)
