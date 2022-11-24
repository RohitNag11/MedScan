import matplotlib.pyplot as plt
import numpy as np
from stl.mesh import Mesh
from mpl_toolkits import mplot3d
import matplotlib

# # Create a new plot
# figure = plt.figure()
# axes = mplot3d.Axes3D(figure)

# # Load the STL files and add the vectors to the plot
# your_mesh = mesh.Mesh.from_file(
#     '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl')
# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# # Auto scale to the mesh size
# scale = your_mesh.points.flatten()
# axes.auto_scale_xyz(scale, scale, scale)

# # Show the plot to the screen
# # plt.savefig('myfilename.png')
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# x = np.linspace(-2, 2, 60)
# y = np.linspace(-2, 2, 60)
# x, y = np.meshgrid(x, y)
# r = np.sqrt(x**2 + y**2)
# z = np.cos(r)
# surf = ax.plot_surface(x, y, z, rstride=2, cstride=2,
#                        cmap='viridis', linewidth=0)

ss = 0.6

mesh = Mesh.from_file(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl')
min_z, max_z = np.min(mesh.z), np.max(mesh.z)
print(f'min_z, max_z: {min_z, max_z}')
height = max_z - min_z
no_slices = np.floor(height / ss) + 1
print(f'no_slices: {no_slices}')
# mesh.translate(np.array([0, 0, -np.min(mesh.z)]))
knee_loc = np.max(mesh.z)
print(f'knee_loc: {knee_loc}')
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
scale = mesh.points.flatten()
ax.auto_scale_xyz(scale, scale, scale)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
