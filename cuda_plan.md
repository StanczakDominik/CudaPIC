# Gryplan

* Create particle data: **N particles**
   * Allocate space for x, y, z, vx, vy, vz arrays
   * Initialize with uniform spatial distribution, zero velocities
* Write a 3d pusher
   * use Boris code
   * set B to 0
* Interpolate grid fields
   * use premade function of X, Y, Z for now
* Visualize results
   * browse through Nvidia samples



Hell, maybe just put all the operations for each iteration in one kernel and have the main loop call that? I'm repeating my code a lot. This seems bad.
What I'm going to do ultimately is switch like this:
particle kernel: update positions, interpolate fields, update velocities, calculate particle indices
* Use particle indices to calculate particle density 
grid kernel: calculate potential (potentially via Fourier transform), calculate field


The grid array can and should be 3d!!!!!!