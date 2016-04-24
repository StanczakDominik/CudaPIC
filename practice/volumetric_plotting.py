import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as mlab
# import sip
# sip.setapi('QString', 2)
N = 20
x = np.linspace(0,1,N, endpoint = False)
y = np.linspace(0,1,N, endpoint = False)
z = np.linspace(0,1,N, endpoint = False)

# x, y, z = np.meshgrid(x,y,z, indexing='ij')
x, y, z = np.mgrid[0:1:20j, 0:1:20j, 0:1:20j]

def plot_contours_potential():
    potential = np.sin(2*np.pi*(x**2+y*z))

    print("Calculated")
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(X, Y, Z, c=potential)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # fig.colorbar(scatter)
    # plt.show()
    source = mlab.pipeline.scalar_field(potential)
    # mlab.pipeline.volume(source, vmax=potential.max()*0.9, vmin=potential.min()*0.8)
    # mlab.pipeline.iso_surface(source, contours=[potential.min()+0.1*potential.ptp(), ])
    # mlab.pipeline.iso_surface(source, contours=[potential.max()-0.1*potential.ptp(), ])
    mlab.pipeline.image_plane_widget(source, plane_orientation='x_axes', slice_index = 0)
    mlab.pipeline.image_plane_widget(source, plane_orientation='y_axes', slice_index = 0)
    mlab.pipeline.image_plane_widget(source, plane_orientation='z_axes', slice_index = 0)
    mlab.pipeline.contour_surface(source, transparent=True, opacity=0.7)
    mlab.show()



def plot_vector_field():

    u =    np.sin(np.pi*x) * np.cos(np.pi*z)
    v = -2*np.sin(np.pi*y) * np.cos(2*np.pi*z)
    w = np.cos(np.pi*x)*np.sin(np.pi*z) + np.cos(np.pi*y)*np.sin(2*np.pi*z)
    print("showing")
    src = mlab.pipeline.vector_field(u, v, w)
    # mlab.pipeline.vectors(src, mask_points=1, scale_factor=3.)
    # magnitude = mlab.pipeline.extract_vector_norm(src)
    # mlab.pipeline.iso_surface(magnitude, contours = [1.9, 0.5])
    mlab.flow(u,v,w)
    # mlab.outline()
    mlab.show()

plot_vector_field()
