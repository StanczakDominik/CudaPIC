import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

I = range(0, 10, 1)
names = ["electrons_positions_{}", "ions_positions_{}"]
labels = ["Electrons", "Electrons"]
colors = ["g", "b"]

fig = plt.figure(figsize=(16,12))

ax_z = fig.add_subplot(242)
ax_x = fig.add_subplot(245)
ax_y = fig.add_subplot(246)

ax_pz = fig.add_subplot(244)
ax_px = fig.add_subplot(247)
ax_py = fig.add_subplot(248)

title = fig.suptitle("iteration: 0")
Xphases = []
Yphases = []
Zphases = []

for ind, name in enumerate(names):
    dataset_name = "data/" + name.format(0) + ".dat"
    data = np.loadtxt(dataset_name)
    # print(data[:10])
    x, y, z, vx, vy, vz = data.T

    ax_x.hist(x, 50, color=colors[ind], label=dataset_name, lw=0, alpha=0.5)
    ax_y.hist(y, 50, color=colors[ind], label=dataset_name, lw=0, alpha=0.5)
    ax_z.hist(z, 50, color=colors[ind], label=dataset_name, lw=0, alpha=0.5)

    ax_px.plot(x, vx, "o", alpha=0.9, c=colors[ind], label=labels[ind])
    ax_py.plot(y, vy, "o", alpha=0.9, c=colors[ind], label=labels[ind])
    ax_pz.plot(z, vz, "o", alpha=0.9, c=colors[ind], label=labels[ind])

ax_x.legend(loc='best')
ax_px.legend(loc='best')

ax_px.set_xlabel('x')
ax_px.set_ylabel('vx')
ax_py.set_xlabel('y')
ax_py.set_ylabel('vy')
ax_pz.set_xlabel('z')
ax_pz.set_ylabel('vz')

ax_x.set_xlabel("x")
ax_y.set_xlabel("y")
ax_z.set_xlabel("z")
def animate(i):
    title.set_text("Iteration: {}".format(i))
    ax_x.cla()
    ax_y.cla()
    ax_z.cla()
    ax_pz.cla()
    ax_py.cla()
    ax_px.cla()
    for ind, name in enumerate(names):
        dataset_name = "data/" + name.format(i) + ".dat"
        print(i, dataset_name)
        data = np.loadtxt(dataset_name)
        x, y, z, vx, vy, vz = data.T

        ax_x.hist(x, 50, color=colors[ind], label=labels[ind], lw=0, alpha=0.5)
        ax_y.hist(y, 50, color=colors[ind], label=labels[ind], lw=0, alpha=0.5)
        ax_z.hist(z, 50, color=colors[ind], label=labels[ind], lw=0, alpha=0.5)

        ax_px.plot(x, vx, "o", alpha=0.9, c=colors[ind], label=labels[ind])
        ax_py.plot(y, vy, "o", alpha=0.9, c=colors[ind], label=labels[ind])
        ax_pz.plot(z, vz, "o", alpha=0.9, c=colors[ind], label=labels[ind])
    return [title, ax_x, ax_y, ax_z, ax_px, ax_py, ax_pz]

anim = animation.FuncAnimation(fig, animate, I)
anim.save('anim_test.mp4', fps=10, dpi=100, writer='ffmpeg', bitrate=1000, extra_args=['-pix_fmt', 'yuv420p'])
plt.close(fig)
