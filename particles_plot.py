import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

I = range(0, 10000, 50)
names = ["electrons_positions_{}", "ions_positions_{}"]
colors = ["g", "b"]

fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(241, projection="3d")
ax_z = fig.add_subplot(242)
ax_x = fig.add_subplot(245)
ax_y = fig.add_subplot(246)


ax_pz = fig.add_subplot(244)
ax_px = fig.add_subplot(247)
ax_py = fig.add_subplot(248)



# figManager = plt.get_current_fig_manager()

allax = [ax, ax_z, ax_x, ax_y, ax_pz, ax_px, ax_py]

for i in I:
    fig.suptitle("iteration: {}".format(i))
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

        ax_px.scatter(x, vx, alpha=0.9, c=colors[ind], label=dataset_name)
        ax_py.scatter(y, vy, alpha=0.9, c=colors[ind], label=dataset_name)
        ax_pz.scatter(z, vz, alpha=0.9, c=colors[ind], label=dataset_name)
    ax.legend(loc='best')
    ax_x.legend(loc='best')
    # figManager.window.showMaximized()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax_px.set_xlabel('x')
    ax_px.set_ylabel('vx')
    ax_py.set_xlabel('y')
    ax_py.set_ylabel('vy')
    ax_pz.set_xlabel('z')
    ax_pz.set_ylabel('vz')

    ax_x.set_xlabel("x")
    ax_y.set_xlabel("y")
    ax_z.set_xlabel("z")

    fig.savefig("data/snap_{}.png".format(i))
    # plt.show()
    for a in allax:
        a.cla()
    # plt.close(fig)
