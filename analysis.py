import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
import h5py
import argparse
from mpl_toolkits.mplot3d import Axes3D

filename = "data.hdf5"
parser = argparse.ArgumentParser()
parser.add_argument("name", help="run name - name of hdf5 group")
args = parser.parse_args()

N_grid = 16
dx = 1/(N_grid-1)

def scalar_field_plot(f, z):
    plt.imshow(f[:,:,z])
    plt.colorbar()
    plt.show()
def particle__plot_onetime(f,skip_N):
    x = f[:,0]
    y = f[:,1]
    z = f[:,2]

    fig1, (axes1, axes2, axes3) = plt.subplots(3)
    axes1.hist(x, bins=N_grid, linewidth=1)
    axes1.set_xlabel("x")
    axes2.hist(y, bins=N_grid, linewidth=1)
    axes2.set_xlabel("y")
    axes3.hist(z, bins=N_grid, linewidth=1)
    axes2.set_xlabel("z")

    fig2, axes_3d = plt.subplots(projection='3d')
    axes_3d.plot(x,y,z,"go")
    plt.show()

def particle_plot_trajectory(f):
    x = f[:,::3]
    y = f[:,1::3]
    z = f[:,2::3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(x.shape[1]):
        ax.plot(x[:,i],y[:,i],z[:,i])
    plt.show()


if __name__=="__main__":
    with h5py.File(filename) as f:
        if args.name not in f:
            grp = f.create_group(args.name)
            for dataset in ("init_density", "final_density"):
                grp[dataset] = np.loadtxt(dataset + ".dat").reshape((N_grid,N_grid,N_grid))
            for dataset in ("trajectory", "init_position", "final_position"):
                grp[dataset] = np.loadtxt(dataset + ".dat")
        else:
            grp = f[args.name]
