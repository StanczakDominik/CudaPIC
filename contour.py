from mayavi import mlab
import numpy as np

N_grid = 8
datafile = np.loadtxt("initial_density.dat")
rho = datafile[:,0].reshape((N_grid,N_grid,N_grid))
Ex = datafile[:,1].reshape((N_grid,N_grid,N_grid))
Ey = datafile[:,2].reshape((N_grid,N_grid,N_grid))
Ez = datafile[:,3].reshape((N_grid,N_grid,N_grid))


mlab.contour3d(rho)
mlab.show()

datafile = np.loadtxt("data/running_density_1.dat")
rho = datafile[:,0].reshape((N_grid,N_grid,N_grid))
Ex = datafile[:,1].reshape((N_grid,N_grid,N_grid))
Ey = datafile[:,2].reshape((N_grid,N_grid,N_grid))
Ez = datafile[:,3].reshape((N_grid,N_grid,N_grid))

mlab.contour3d(rho)
mlab.show()
