import trimesh
import numpy as np
from matplotlib.patches import Polygon
import pydicom
import numpy as np
import os


class BoneMesh:
    def __init__(self, path: str):
        self.mesh = self.__read_stl(path)
        self.z_bounds = self.__get_z_bounds()
        return None

    def __read_stl(self, path: str):
        mesh = trimesh.load_mesh(path, file_type='stl')
        return mesh

    def __get_z_bounds(self) -> list[float]:
        min_z = min(self.mesh.vertices[:, 2])
        max_z = max(self.mesh.vertices[:, 2])
        return [min_z, max_z]

    def get_z_section_points(self, z: float):
        section = self.mesh.section(
            plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        order = section.vertex_nodes[:, 0]
        raw_vertices = section.vertices[:, :2]
        vertices = [raw_vertices[i] for i in order]
        if section:
            return np.array(vertices)
        else:
            return np.array([0, 0])

    def get_z_section_polygon(self, z: float, label='bone section', ec='r'):
        z_lt_section_points = self.get_z_section_points(z)
        return Polygon(z_lt_section_points, label=label, ec=ec, fc=ec+'40', lw=2)


class DicomCT:
    def __init__(self, path: str):
        self.axial_slices = self.__get_planar_slices(path, (1, 0, 0, 0, 1, 0))
        self.ni, self.nj, self.nk = self.axial_slices[0].pixel_array.shape + (len(
            self.axial_slices),)
        self.dx, self.dy, self.dz = self.__get_spacing()
        self.x_bounds, self.y_bounds, self.z_bounds = self.__get_bounding_box()
        return None

    def __get_planar_slices(self,
                            path: str,
                            plane_orientation: tuple[int, int, int, int, int, int]):
        slices = []
        with os.scandir(path) as folder:
            for entry in folder:
                if entry.is_file() and entry.name != 'VERSION':
                    # print(f"loading: {entry.path}")
                    dcm = pydicom.dcmread(entry.path)
                    if (hasattr(dcm, 'SliceLocation')
                            and dcm.ImageOrientationPatient == list(plane_orientation)):
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

    def get_k_index(self, z: float):
        k = (self.nk - 1) * \
            (z - self.z_bounds[0]) / (self.z_bounds[1] - self.z_bounds[0])
        return int(k)

    def get_z_pos(self, k: int) -> float:
        return self.axial_slices[k].ImagePositionPatient[2]

    def get_i_index(self, x: float):
        i = (self.ni - 1) * \
            (x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0])
        return int(i)

    def get_j_index(self, y: float):
        j = (self.ni - 1) * \
            (y - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0])
        return int(j)

    def get_k_image(self, k: int):
        raw_img = self.axial_slices[k].pixel_array
        return np.flipud(raw_img)

    def get_z_image(self, z: float):
        k = self.get_k_index(z)
        return self.get_k_image(k)
