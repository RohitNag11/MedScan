import medscan.readers as msr
import cv2
import numpy as np

import matplotlib.pyplot as plt


class SoftTissueSegmenter:
    def __init__(self, body_CT: msr.DicomCT):
        self.body_CT = body_CT
        self.segmented_slices = {}

    def addBoneMeshSlices(self, bone_mesh: msr.BoneMesh):
        '''Populates segmented slices for a particular bone as an array of dictionaries: 
        [{k0: k0_img},..,{kn-1: kn-1_img}]'''
        segmented_slices = []
        for k, slice in enumerate(self.body_CT.axial_slices):
            segmented_image = self.__getSegmentedImage(bone_mesh, k)
            if segmented_image is not None:
                segmented_slices.append({k: segmented_image})
        self.segmented_slices[bone_mesh.name] = segmented_slices

    def __getSegmentedImage(self, bone_mesh: msr.BoneMesh, k: int):
        z = self.body_CT.get_z_pos(k)
        ct_section_img = self.body_CT.get_k_image(k)
        if bone_mesh.z_bounds[0] <= z <= bone_mesh.z_bounds[1]:
            bone_section_poly_points = bone_mesh.get_z_section_points(z)
            bone_section_poly_pixels = self.__translateToPixelSpace(
                bone_section_poly_points)
            bone_section_img = self.__getPolyImage(bone_section_poly_pixels)
            return np.multiply(ct_section_img, bone_section_img)
        return None

    def __translateToPixelSpace(self, cartesian_points):
        x0, xn = self.body_CT.x_bounds
        y0, yn = self.body_CT.y_bounds
        cartesian_points[:, 0] -= x0
        cartesian_points[:, 1] -= yn
        T = np.array([[(self.body_CT.ni - 1) / (xn - x0), 0],
                      [0, (self.body_CT.nj - 1) / (y0 - yn)]])
        return np.int32(np.matmul(cartesian_points, T))

    def __getPolyImage(self, poly_pixels):
        img_dim = (self.body_CT.ni, self.body_CT.nj)
        img = np.zeros(img_dim)
        cv2.fillPoly(img, pts=[poly_pixels], color=(255, 255, 255))
        return img
