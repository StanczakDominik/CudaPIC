from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt

N_grid = 16
NT = 100
datafile = np.loadtxt("running_density.dat")
for i in range(NT):
	rho = datafile[NT*i:NT*(i+1),0].reshape((N_grid,N_grid,N_grid))
	Ex = datafile[NT*i:NT*(i+1),1].reshape((N_grid,N_grid,N_grid))
	Ey = datafile[NT*i:NT*(i+1),2].reshape((N_grid,N_grid,N_grid))
	Ez = datafile[NT*i:NT*(i+1),3].reshape((N_grid,N_grid,N_grid))
	mlab.contour3d(rho)
	mlab.savefig("frame_{}.png".format(i))
	mlab.clf()
	