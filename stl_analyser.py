import matplotlib.pyplot as plt
import numpy
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib
print(matplotlib.get_backend())

# Create a new plot
figure = plt.figure()
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
plt.savefig('myfilename.png')
plt.show()
