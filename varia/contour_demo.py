#!/usr/bin/python2
from mayavi import mlab
import numpy as np
import os

N_grid = 16
N_grid_all = N_grid**3
NT = 1000
filename_template = "gfx/running_density_{}.dat"
filenames = [filename_template.format(i) for i in range(NT)]

if not os.path.isfile("gfx/params.txt"):
	minimum = 1000
	maximum = -1000
	for filename in filenames:
		rho = np.loadtxt(filename)[:,0]
		if rho.min() < minimum:
			minimum = rho.min()
		if rho.max() > maximum:
			maximum = rho.max()
	ptp = maximum - minimum
	print(minimum, maximum, ptp)
	params = np.array([minimum, maximum, ptp])
	np.savetxt("gfx/params.txt", params)
else:
	params = np.loadtxt("gfx/params.txt")
	minimum, maximum, ptp = params

maximum = 0.8
minimum = -maximum
ptp = 2*maximum

frame_filename = "gfx/frame_{}.png"
f = mlab.figure(size=(1024,768), bgcolor=(0.0,0.0,0.0))
for i, filename in enumerate(filenames):
	datafile = np.loadtxt(filename)
	print(str(i) + "\r")
	rho = datafile[:,0].reshape((N_grid,N_grid,N_grid))
	minimum, maximum, ptp = rho.min(), rho.max(), rho.ptp()
	# # Ex = datafile[:,1].reshape((N_grid,N_grid,N_grid))
	# # Ey = datafile[:,2].reshape((N_grid,N_grid,N_grid))
	# # Ez = datafile[:,3].reshape((N_grid,N_grid,N_grid))
	# contours = np.linspace(minimum+0.1*ptp, maximum-0.1*ptp, 5)
	obj = mlab.contour3d(rho, figure=f, contours=10, vmin = minimum, vmax = maximum, opacity=0.3)#, contours=10)
	mlab.colorbar(obj)
	# mlab.pipeline.volume(mlab.pipeline.scalar_field(rho), figure=f, vmin=minimum + 0.2*datafile.ptp(), vmax = minimum + 0.8*datafile.ptp())
	mlab.savefig(frame_filename.format(i), figure=f)
	mlab.clf(f)
	# mlab.show()
