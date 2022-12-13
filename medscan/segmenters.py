import medscan.readers as msr
import cv2
import numpy as np
import matplotlib.pyplot as plt


class SoftTissueSegmenter:
    def __init__(self, body_CT: msr.DicomCT):
        self.body_CT = body_CT
        self.segmented_slices = {}
        self.segmented_point_cloud = {}

    def add_bone_mesh(self, bone_mesh: msr.BoneMesh):
        '''Populates segmented slices for a particular bone as an array of tuples: 
        [(k0: k0_img),..,(kn-1: kn-1_img)]'''
        segmented_slices = []
        point_cloud = []
        for k, slice in enumerate(self.body_CT.axial_slices):
            z = self.body_CT.get_z_pos(k)
            segmented_image = self.__get_segmented_image(bone_mesh, k)
            if segmented_image is not None:
                segmented_slices.append((k, segmented_image))
                for j, row in enumerate(segmented_image):
                    y = j * self.body_CT.dy
                    for i, pixel in enumerate(row):
                        x = i * self.body_CT.dx
                        if pixel:
                            point_cloud.append([x, y, z, pixel])
        self.segmented_slices[bone_mesh.name] = segmented_slices
        self.segmented_point_cloud[bone_mesh.name] = np.array(point_cloud)

    def __get_segmented_image(self, bone_mesh: msr.BoneMesh, k: int):
        z = self.body_CT.get_z_pos(k)
        ct_section_img = self.body_CT.get_k_image(k)
        if bone_mesh.z_bounds[0] <= z <= bone_mesh.z_bounds[1]:
            bone_section_poly_points = bone_mesh.get_z_section_points(z)
            bone_section_poly_pixels = self.__translate_to_pixel_space(
                bone_section_poly_points)
            bone_section_img = self.__get_poly_image(bone_section_poly_pixels)
            return np.multiply(ct_section_img, bone_section_img)
        return None

    def __translate_to_pixel_space(self, cartesian_points):
        x0, xn = self.body_CT.x_bounds
        y0, yn = self.body_CT.y_bounds
        cartesian_points[:, 0] -= x0
        cartesian_points[:, 1] -= yn
        T = np.array([[(self.body_CT.ni - 1) / (xn - x0), 0],
                      [0, (self.body_CT.nj - 1) / (y0 - yn)]])
        return np.int32(np.matmul(cartesian_points, T))

    def __get_poly_image(self, poly_pixels):
        img_dim = (self.body_CT.ni, self.body_CT.nj)
        img = np.zeros(img_dim)
        cv2.fillPoly(img, pts=[poly_pixels], color=(255, 255, 255))
        return img

    def get_down_sampled_point_cloud(self, bone_mesh, pixels=10):
        # down_sampled_img_3d = self.__get_down_sampled_slices(bone_mesh, pixels)
        down_sampled_img_3d = self.__get_down_sample_img_3d(bone_mesh, pixels)
        point_cloud = []
        for k, slice in enumerate(down_sampled_img_3d):
            z = self.body_CT.get_z_pos(k*pixels)
            for j, row in enumerate(slice):
                y = pixels * j * self.body_CT.dy
                for i, pixel in enumerate(row):
                    x = pixels * i * self.body_CT.dx
                    if pixel:
                        point_cloud.append([x, y, z, pixel])
        return np.array(point_cloud)

    def __get_down_sample_img_3d(self, bone_mesh, pixels):
        img_3d = self.__get_down_sampled_slices(bone_mesh, pixels)
        first_slice = img_3d[0]
        down_sampled_img_3d = np.zeros(first_slice.shape).tolist()
        for j, row in enumerate(first_slice):
            for i in range(len(row)):
                down_sampled_img_3d[i][j] = self.__get_down_sampled_pixel_row(
                    img_3d[:, i, j], pixels)
        return np.array(down_sampled_img_3d).T

    def __get_down_sampled_slices(self, bone_mesh, pixels):
        slices = self.segmented_slices[bone_mesh.name]
        img_3d = np.array([self.__get_down_sampled_image(
            slice[1], pixels) for slice in slices])
        return img_3d

    def __get_down_sampled_image(self, slice, pixels):
        down_sampled_rows = np.array(
            [self.__get_down_sampled_pixel_row(row, pixels) for row in slice])
        down_sampled_columns = np.array(
            [self.__get_down_sampled_pixel_row(row, pixels) for row in down_sampled_rows.T])
        return down_sampled_columns.T

    def __get_down_sampled_pixel_row(self, row, pixels):
        down_sampled_row = []
        for i in range(0, len(row), pixels):
            down_sampled_row.append(np.mean(row[i:i+pixels]))
        return np.array(down_sampled_row)
