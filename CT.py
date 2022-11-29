import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import cv2 as cv
from matplotlib.patches import Polygon


class CT:
    def __init__(self, path):
        self.axial_slices = self.__get_planar_slices(path, [1, 0, 0, 0, 1, 0])
        self.ni, self.nj, self.nk = self.axial_slices[0].pixel_array.shape + (len(
            self.axial_slices),)
        self.dx, self.dy, self.dz = self.__get_spacing()
        self.x_bounds, self.y_bounds, self.z_bounds = self.__get_bounding_box()

    def __get_planar_slices(self, path, plane_orientation):
        slices = []
        with os.scandir(path) as folder:
            for entry in folder:
                if entry.is_file() and entry.name != 'VERSION':
                    # print(f"loading: {entry.path}")
                    dcm = pydicom.dcmread(entry.path)
                    if (hasattr(dcm, 'SliceLocation')
                            and dcm.ImageOrientationPatient == plane_orientation):
                        slices.append(dcm)
        return sorted(slices, key=lambda s: s.SliceLocation)

    def __get_spacing(self):
        dy, dx = self.axial_slices[0].PixelSpacing
        dz = self.axial_slices[0].SliceThickness
        return [dx, dy, dz]

    def __get_bounding_box(self):
        min_x, min_y = self.axial_slices[0].ImagePositionPatient[:2]
        max_x, max_y = min_x + (self.ni - 1) * \
            self.dx, min_y + (self.nj - 1) * self.dy
        min_z, max_z = self.axial_slices[0].SliceLocation, self.axial_slices[-1].SliceLocation
        return np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])

    def get_k_index(self, z):
        k = (self.nk - 1) * \
            (z - self.z_bounds[0]) / (self.z_bounds[1] - self.z_bounds[0])
        return int(k)

    def get_z_pos(self, k):
        return self.axial_slices[k].ImagePositionPatient[2]

    def get_i_index(self, x):
        i = (self.ni - 1) * \
            (x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0])
        return int(i)

    def get_j_index(self, y):
        j = (self.ni - 1) * \
            (y - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0])
        return int(j)

    def get_k_image(self, k):
        raw_img = self.axial_slices[k].pixel_array
        return np.flipud(raw_img)

    def get_z_image(self, z):
        k = self.get_k_index(z)
        return self.get_k_image(k)


class BoneMesh:
    def __init__(self, path):
        self.mesh = self.__read_stl(path)
        return None

    def __read_stl(self, path):
        mesh = trimesh.load_mesh(path, file_type='stl')
        return mesh

    def get_z_section_points(self, z):
        section = self.mesh.section(
            plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        order = section.vertex_nodes[:, 0]
        raw_vertices = section.vertices[:, :2]
        vertices = [raw_vertices[i] for i in order]
        if section:
            return np.array(vertices)
        else:
            return [0, 0]

    def get_z_section_polygon(self, z, label, ec='r', fc='#FFFFFF4A'):
        z_lt_section_points = self.get_z_section_points(z)
        return Polygon(z_lt_section_points, label=label, edgecolor=ec, facecolor=fc)


body_CT = CT(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_MJM010_Phantom1607,2003840n/R4G1B43W/TDS102WE')
lt_bone_mesh = BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl')
rt_bone_mesh = BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM10_2003840N_Right Tibia.stl')
k = 150
z = body_CT.get_z_pos(k)
z_img = body_CT.get_z_image(z)
z_lt_poly = lt_bone_mesh.get_z_section_polygon(z, 'left tibia', 'r')
z_rt_poly = rt_bone_mesh.get_z_section_polygon(z, 'right tibia', 'b')
fig, ax = plt.subplots()
ax.set_title('Axial Segmentation of Bones from CT Scan')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.imshow(z_img, extent=np.array(
    [body_CT.x_bounds, body_CT.y_bounds]).flatten())
ax.add_patch(z_lt_poly)
ax.add_patch(z_rt_poly)
ax.legend()
plt.show()
# cv.imshow(f'ct image z={z}', a*255)
# cv.waitKey(0)
