import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as ani
import h5py
import argparse
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'

filename = "data.hdf5"
parser = argparse.ArgumentParser()
parser.add_argument("name", help="run name - name of hdf5 group")
args = parser.parse_args()

N_grid = 16
dx = 1/(N_grid-1)

def animate_scalar_field(f):
    scalar_field = f[...]
    fig, axes = plt.subplots()
    IM = axes.imshow(scalar_field[:,:,0], origin='upper', vmin=np.min(scalar_field), vmax=np.max(scalar_field), extent=(0,N_grid*dx, 0, N_grid*dx))
    axes.set_title(u"Przekrój w płaszczyźnie z")
    text = axes.text(2*dx, 2*dx, "z = {}".format(0), color=(0,0,0,1),fontsize=18)
    axes.set_xlabel("x")
    axes.set_ylabel("y")

    def init():
        text.set_text("")
        IM.set_array(scalar_field[:,:,0])
        return IM, text
    def animate(i):
        IM.set_array(scalar_field[:,:,i])
        text.set_text("z = {}".format(i))
        return IM, text

    fig.colorbar(IM, orientation="vertical")
    anim = ani.FuncAnimation(fig, animate, interval=1000, frames=N_grid, blit=True, init_func=init)
    # if save:
    #     anim.save("grafika/animation_scalar_field.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()

def scalar_field_plot(f, z):
    minv = f[...].min()
    maxv = f[...].max()
    plt.imshow(f[:,:,z], vmin = minv, vmax=maxv, interpolation='nearest')
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
                filename = dataset + ".dat"
                if os.path.isfile(filename):
                    grp[dataset] = np.loadtxt(filename).reshape((N_grid,N_grid,N_grid))
            for dataset in ("trajectory", "init_position", "final_position"):
                filename = dataset + ".dat"
                if os.path.isfile(filename):
                    grp[dataset] = np.loadtxt(filename)
        else:
            grp = f[args.name]
        for i in grp:
            print(i)
        animate_scalar_field(grp['init_density'])
