import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np
import os
from .helpers import geometry as geom
import time
from collections import Counter


class BoneMesh:
    def __init__(self, path: str, name: str):
        self.name = name
        self.mesh = self.__read_stl(path)
        self.x_bounds, self.y_bounds, self.z_bounds = self.__get_bounds()
        self.x_mid_plane = np.mean(self.x_bounds)
        self.y_mid_plane = np.mean(self.y_bounds)
        return None

    def __read_stl(self, path: str) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(path, file_type='stl')
        return mesh

    def __get_bounds(self):
        x_bounds, y_bounds, z_bounds = self.mesh.bounding_box.bounds.T
        return x_bounds, y_bounds, z_bounds

    def get_z_section_points(self, z: float):
        section = self.mesh.section(
            plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section:
            order = section.vertex_nodes[:, 0]
            raw_vertices = section.vertices[:, :2]
            ordered_vertices = [raw_vertices[i] for i in order]
            return np.array(ordered_vertices)
        else:
            return np.array([[0, 0], [0, 0]])

    def get_z_section_polygon(self, z: float, label='bone section', ec='r'):
        z_lt_section_points = self.get_z_section_points(z)
        return Polygon(z_lt_section_points, label=label, ec=ec, fc=ec+'40', lw=2)


class DicomCT:
    def __init__(self, dicomdir_path: str):
        image_paths = self.__get_image_paths_from_dicomdir(dicomdir_path)
        self.axial_slices = self.__get_planar_slices(
            image_paths, [1, 0, 0, 0, 1, 0])
        self.img3d = np.array(
            [slice.pixel_array for slice in self.axial_slices])
        self.ni, self.nj, self.nk = self.axial_slices[0].pixel_array.shape + (len(
            self.axial_slices),)
        self.dx, self.dy, self.dz = self.__get_spacing()
        self.x_bounds, self.y_bounds, self.z_bounds = self.__get_bounding_box()
        self.z_values = self.__get_z_values()
        self.x_mid_plane = self.__get_x_mid_plane()
        return None

    def __get_z_values(self):
        return np.linspace(self.z_bounds[0], self.z_bounds[1], self.nk)

    def __get_x_mid_plane(self):
        return sum(self.x_bounds) / 2

    def __get_image_paths_from_dicomdir(self, dicom_path: str):
        dicomdir_path = os.path.relpath(f'{dicom_path}/DICOMDIR', os.getcwd())
        ds = pydicom.dcmread(dicomdir_path, stop_before_pixels=True)
        # select the first patient: likely only one in the directory
        patient = ds.patient_records[0]
        # Find the first STUDY record for the patient
        study = next(
            (ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"), None)
        # Find the series with the most scan instances
        most_images = 0
        selected_images = None
        # Iterate through the children of the study and find the series with the most scan instances
        for record in study.children:
            if record.DirectoryRecordType != "SERIES":
                continue
            # Find all the IMAGE records in the series
            images = [
                ii for ii in record.children if ii.DirectoryRecordType == "IMAGE"]
            if len(images) > most_images:
                most_images = len(images)
                selected_images = images
        # Each IMAGE contains a relative file path to the root directory
        paths = [record["ReferencedFileID"].value for record in selected_images]
        return [os.path.join(dicom_path, *p) for p in paths]

    def __get_planar_slices(self,
                            image_paths: list,
                            plane_orientation: list[int, int, int, int, int, int]) -> list[pydicom.FileDataset]:
        slices = []
        filtered_paths = [path for path in image_paths if pydicom.dcmread(
            path, stop_before_pixels=True).ImageOrientationPatient == plane_orientation]
        for path in filtered_paths:
            dcm = pydicom.dcmread(path)
            if 'SliceLocation' in dcm:
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

    def get_i_index(self, x: float):
        i = (self.ni - 1) * \
            (x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0])
        return int(i)

    def get_j_index(self, y: float):
        j = (self.ni - 1) * \
            (y - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0])
        return int(j)

    def get_k_index(self, z: float):
        k = (self.nk - 1) * \
            (z - self.z_bounds[0]) / (self.z_bounds[1] - self.z_bounds[0])
        return int(k)

    def get_x_pos(self, i: int) -> float:
        x0, xn = self.x_bounds
        return i * (xn - x0) / (self.ni - 1)

    def get_y_pos(self, j: int) -> float:
        y0, yn = self.y_bounds
        return j * (y0 - yn) / (self.nj - 1)

    def get_z_pos(self, k: int) -> float:
        return self.axial_slices[k].ImagePositionPatient[2]

    def get_k_image(self, k: int, callibrate: bool = False):
        raw_img = self.axial_slices[k].pixel_array
        # calibrate immage to densities
        hu_img = apply_modality_lut(
            raw_img, self.axial_slices[k]) if callibrate else raw_img
        return np.flipud(hu_img)

    def get_z_image(self, z: float, callibrate: bool = False):
        k = self.get_k_index(z)
        return self.get_k_image(k, callibrate)

    def calibrate_image_by_k(self, img, k):
        return apply_modality_lut(img, self.axial_slices[k])


class BoneCT:
    def __init__(self, body_ct: DicomCT, bone_mesh: BoneMesh, roi_depth: float = 20.0, filter_percent: float = 30.0):
        self.body_ct = body_ct
        self.bone_mesh = bone_mesh
        self.roi_depth = roi_depth
        self.side = self.__get_side()
        self.slices = self.__get_slices()
        self.img_3d = self.__get_img_3d()
        # start_time = time.time()
        self.all_points_4d = self.__get_all_points_4d()
        # print("Time taken: ", time.time() - start_time, "seconds")
        self.medial_points_4d = self.__get_medial_points_4d()
        self.implant_roi_points_4d = self.__get_implant_roi_points_4d(
            filter_percent)
        return None

    def __get_side(self):
        '''Returns the side ('left' or 'right) of the body the bone is on'''
        if self.bone_mesh.x_mid_plane > self.body_ct.x_mid_plane:
            return 'left'
        return 'right'

    def __get_slices(self):
        '''Returns a list of slices that contain the bone as an array of tuples: 
        [(z0: z0_img),..,(kn-1: kn-1_img)].'''
        slices = []
        for k, slice in enumerate(self.body_ct.axial_slices):
            z = self.body_ct.get_z_pos(k)
            if self.bone_mesh.z_bounds[0] <= z <= self.bone_mesh.z_bounds[1]:
                ct_section_img = self.body_ct.get_k_image(k, callibrate=True)
                bone_section_poly_points = self.bone_mesh.get_z_section_points(
                    z)
                bone_section_poly_pixels = geom.cartesian_2d_to_pixel_space(
                    bone_section_poly_points,
                    self.body_ct.x_bounds,
                    self.body_ct.y_bounds,
                    self.body_ct.ni,
                    self.body_ct.nj)
                bone_section_img = geom.get_poly_image(
                    bone_section_poly_pixels,
                    self.body_ct.ni,
                    self.body_ct.nj)
                segmented_image = np.multiply(ct_section_img, bone_section_img)
                # calibrated_segmented_image = self.body_ct.calibrate_image_by_k(
                #     segmented_image, k)
                slices.append((z, segmented_image))
        return slices

    def __get_img_3d(self):
        images = [img for z, img in self.slices]
        return np.stack(images, axis=0)

    def __get_all_points_4d(self):
        '''Returns a 4d point cloud of the bone:
        (x, y, z, density)'''
        k, j, i = np.where(self.img_3d > 0)
        x_bounds, y_bounds = self.body_ct.x_bounds, self.body_ct.y_bounds
        z_bounds = self.bone_mesh.z_bounds
        x = x_bounds[0] + (x_bounds[1] - x_bounds[0]) * \
            i / self.img_3d.shape[2]
        y = y_bounds[1] + (y_bounds[0] - y_bounds[1]) * \
            j / self.img_3d.shape[1]
        z = z_bounds[0] + (z_bounds[1] - z_bounds[0]) * \
            k / self.img_3d.shape[0]
        point_cloud = np.column_stack((x, y, z, self.img_3d[k, j, i]))
        return point_cloud

    def __get_medial_points_4d(self):
        '''Returns a point cloud of the medial side of the bone'''
        mask = (self.all_points_4d[:, 0] < self.bone_mesh.x_mid_plane) if self.side == 'left' else (
            self.all_points_4d[:, 0] > self.bone_mesh.x_mid_plane)
        z_min = self.bone_mesh.z_bounds[1] - self.roi_depth
        mask &= self.all_points_4d[:, 2] >= z_min
        return self.all_points_4d[mask]

    def __get_implant_roi_points_4d(self, filter_percent):
        '''Returns the points within the implant region.'''
        # Project the medial points onto the xz plane
        projected_points_3d = geom.project_points_to_plane(
            self.medial_points_4d[:, :3], np.array([0, 1, 0]))
        top_most_points = self.__get_top_most_projected_points(
            projected_points_3d,
            filter_percent)
        x, z = top_most_points.T
        # fig, ax = plt.subplots()
        # ax.scatter(x, z)
        # plt.show()
        # print(min(z))
        implant_z_plane = self.__get_implant_z_plane(
            top_most_points)
        # Remove 3cm from the bottom of the tibia flat region (to remove cartilage)
        filtered_points = self.medial_points_4d[self.medial_points_4d[:, 2]
                                                < implant_z_plane - 3]
        # if self.side == 'left':
        #     return filtered_points[filtered_points[:, 0] < implant_x_plane]
        # return filtered_points[filtered_points[:, 0] > implant_x_plane]
        return filtered_points

    def __get_top_most_projected_points(self, projected_points_3d, filter_percent):
        '''Returns the top most points in the projected point cloud.'''
        x_to_points = {}
        for point in projected_points_3d:
            x = point[0]
            if x not in x_to_points:
                x_to_points[x] = []
            x_to_points[x].append(point[2])
        top_points = np.array([[x, max(zs)] for x, zs in x_to_points.items()])
        top_points = top_points[top_points[:, 0].argsort()]
        left_index = int(len(top_points[:, 0]) * filter_percent / 200)
        return top_points[left_index:-left_index]

    def __get_implant_z_plane(self, top_most_points):
        '''Returns the x plane of the implant.'''
        x, z = top_most_points.T
        ()
        z_counts = Counter(z)
        most_common_z = max(z, key=z_counts.get)
        # dz = np.abs(np.gradient(z))
        # return x[np.argmax(dz)], z[np.argmax(dz)]
        return most_common_z
