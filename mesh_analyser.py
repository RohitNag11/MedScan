import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon
import trimesh
import numpy as np
import cv2 as cv


# print(dir(trimesh))
path = '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl'

origin = np.array([-190.630859375, -341.130859375, -84])
mesh = trimesh.load_mesh(path, file_type='stl')
mesh.apply_translation(origin * -1)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# trisurf = ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1],
#                           triangles=mesh.faces, Z=mesh.vertices[:, 2], label='Left Tibia')
# ax.set_aspect('equal')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# trisurf._edgecolors2d = trisurf._edgecolor3d
# trisurf._facecolors2d = trisurf._facecolor3d
# ax.legend()

# plt.show()

print(f'min_z = {min(mesh.vertices[:, 2])}')


slice = mesh.section(plane_origin=[0, 0, 200], plane_normal=[0, 0, 1])
# print(slice.vertices)

# # take 2D slice (before was still 3D)
# slice_2D, to_3D = slice.to_planar()
# # get vertices
# vertices = np.asanyarray(slice_2D.vertices)
vertices = slice.vertices[:, :2]
# # vertices = (vertices*10).astype(int)
# # min_x = min(vertices[:, 0])
# # min_y = min(vertices[:, 1])
# # vertices[:, 0] -= min_x
# # vertices[:, 1] -= min_y
# plot
fig, ax = plt.subplots()

x, y = vertices.T
ax.scatter(x, y, s=0.4)
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# # image = np.zeros((max(vertices[:, 0]), max(vertices[:, 1])))
# # cv.fillPoly(image, pts=[vertices], color=(255, 255, 255))
# # cv.imshow("filledPolygon", image)

# # cv.imwrite('image.jpg', image)
