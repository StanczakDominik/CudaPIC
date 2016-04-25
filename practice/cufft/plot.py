import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
N=64
x = np.loadtxt("x.dat").reshape((N,N))
y = np.loadtxt("y.dat").reshape((N,N))
u = np.loadtxt("u.dat").reshape((N,N))
u_a = np.loadtxt("u_a.dat").reshape((N,N))
f = np.loadtxt("f.dat").reshape((N,N))
print(x, y, u)
print((u-u_a).mean(), (u-u_a).shape, (u-u_a).max(), (u-u_a).min(), (u-u_a).std())

fig = plt.figure()
ax = fig.gca(projection='3d')
surf1 = ax.plot_surface(x, y, u, rstride=1, cstride=1,
        cmap=cm.coolwarm, antialiased=False, linewidth=0)
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.set_title("Analytical")
surf2 = ax2.plot_surface(x, y, u_a, rstride=1, cstride=1,
        cmap=cm.coolwarm, antialiased=False, linewidth=0)
# surf2 = ax2.plot_surface(x, y, f, rstride=1, cstride=1,
#         cmap=cm.coolwarm, antialiased=False, linewidth=0)


plt.show()
