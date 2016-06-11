import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

I = range(0, 100000, 100)
names = ["electrons_positions_{}", "ions_positions_{}"]
colors = ["g", "b"]

for i in I:
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(221, projection="3d")
    ax_x = fig.add_subplot(222)
    ax_y = fig.add_subplot(223)
    ax_z = fig.add_subplot(224)

    ax.set_title("iteration: {}".format(i))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    for ind, name in enumerate(names):
        dataset_name = "data/" + name.format(i) + ".dat"
        print(i, dataset_name)
        data = np.loadtxt(dataset_name)
        # print(data[:10])
        x, y, z, vx, vy, vz = data.T
        ax.scatter(x, y, z, alpha=0.9, c=colors[ind], label=dataset_name)

        ax_x.hist(x, 50, color=colors[ind], label=dataset_name, lw=0, alpha=0.5)
        ax_y.hist(y, 50, color=colors[ind], label=dataset_name, lw=0, alpha=0.5)
        ax_z.hist(z, 50, color=colors[ind], label=dataset_name, lw=0, alpha=0.5)
    ax.legend(loc='best')
    ax_x.legend(loc='best')
    ax_x.set_xlabel("x")
    ax_y.set_xlabel("y")
    ax_z.set_xlabel("z")
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.savefig("data/snap_{}.png".format(i))
    plt.show()
    # plt.close(fig)
