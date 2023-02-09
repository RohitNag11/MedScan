import medscan.readers as msr
import cv2
import numpy as np


class SoftTissueSegmenter:
    def __init__(self, body_CT: msr.DicomCT):
        self.body_CT = body_CT
        self.segmented_slices = {}
        self.segmented_point_clouds = {}
        self.medial_point_clouds = {}
        self.sides = {}
        self.implant_roi_point_clouds = {}

    def add_bone_mesh(self, bone_mesh: msr.BoneMesh, roi_depth=30):
        '''Populates segmented slices for a particular bone as an array of tuples: 
        [(k0: k0_img),..,(kn-1: kn-1_img)]'''
        segmented_slices = []
        point_cloud = []
        x_bounds = self.body_CT.x_bounds
        y_bounds = self.body_CT.y_bounds
        for k, slice in enumerate(self.body_CT.axial_slices):
            z = self.body_CT.get_z_pos(k)
            segmented_image = self.__get_segmented_image(bone_mesh, k)
            if segmented_image is not None:
                segmented_slices.append((k, segmented_image))
                # Get the non-zero pixels in the image
                non_zero_pixels = np.transpose(np.nonzero(segmented_image))
                # Map the non-zero pixels to their corresponding x, y coordinates
                x_coords = (non_zero_pixels[:, 1] / segmented_image.shape[1]) * (
                    x_bounds[1] - x_bounds[0]) + x_bounds[0]
                y_coords = (non_zero_pixels[:, 0] / segmented_image.shape[0]) * (
                    y_bounds[0] - y_bounds[1]) + y_bounds[1]
                # Combine x, y, z, and pixel values into a list of tuples
                pixel_cloud = np.column_stack((x_coords, y_coords, np.ones(len(
                    x_coords)) * z, segmented_image[non_zero_pixels[:, 0], non_zero_pixels[:, 1]]))
                point_cloud.extend(pixel_cloud.tolist())
        self.segmented_slices[bone_mesh.name] = segmented_slices
        self.segmented_point_clouds[bone_mesh.name] = np.array(point_cloud)
        self.sides[bone_mesh.name] = self.__get_tibia_side(bone_mesh)
        self.medial_point_clouds[bone_mesh.name] = self.__get_medial_point_cloud(
            bone_mesh, roi_depth)
        self.implant_roi_point_clouds[bone_mesh.name] = self.__get_implant_roi_point_coloud(
            bone_mesh, 25)

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
        cv2.fillPoly(img,
                     pts=[poly_pixels],
                     color=(255, 255, 255))
        return img

    def get_down_sampled_point_cloud(self, bone_mesh: msr.BoneMesh, voxel_size=(5, 5, 5)):
        slices = self.segmented_slices[bone_mesh.name]
        nj, ni = self.body_CT.nj, self.body_CT.ni
        di, dj, dk = voxel_size
        img_3d = np.array([slice[1] for slice in slices])
        k_array = [slice[0] for slice in slices]
        point_cloud = []
        for k in k_array[::dk]:
            for j in range(0, nj, dj):
                for i in range(0, ni, di):
                    slice3d = img_3d[k: k+dk, j: j+dj, i: i+di]
                    if np.count_nonzero(slice3d):
                        shape = np.array(slice3d.shape)
                        centre = np.add((shape - 1) / 2, np.array([k, j, i]))
                        x = self.body_CT.get_x_pos(centre[2])
                        y = self.body_CT.get_y_pos(centre[1])
                        z = self.body_CT.get_z_pos(int(centre[0]))
                        avg = slice3d.sum() / slice3d.size
                        point_cloud.append([x, y, z, avg])
        return np.array(point_cloud)

    def __get_tibia_side(self, bone_mesh: msr.BoneMesh):
        x_body_mid_plane = self.body_CT.x_mid_plane
        x_bone_mid_plane = bone_mesh.x_mid_plane
        if x_bone_mid_plane > x_body_mid_plane:
            return 'left'
        else:
            return 'right'

    def __get_medial_point_cloud(self, bone_mesh: msr.BoneMesh, roi_depth):
        tibia_side = self.sides[bone_mesh.name]
        x_bone_mid_plane = bone_mesh.x_mid_plane
        filtered_cloud = self.segmented_point_clouds[bone_mesh.name]
        # If left tibia:
        if tibia_side == 'left':
            filtered_cloud = filtered_cloud[filtered_cloud[:, 0]
                                            < x_bone_mid_plane]
        # If right tibia:
        else:
            filtered_cloud = filtered_cloud[filtered_cloud[:, 0]
                                            > x_bone_mid_plane]
        z_max = np.max(filtered_cloud[:, 2])
        z_min = z_max - roi_depth
        z_range = [z_min, z_max]
        mask = (filtered_cloud[:, 2] >= z_range[0]) & (
            filtered_cloud[:, 2] <= z_range[1])
        return filtered_cloud[mask]

    def project_3d_to_plane(self, point_cloud_3d, plane_normal, plane_point=np.array([0, 0, 0])):
        """
        Project a 3D point cloud onto a plane.

        Parameters:
        - points: numpy array of shape (N, 3) where N is the number of points
            in the point cloud.
        - plane_normal: numpy array of shape (3,) representing the normal
            vector of the plane.
        - point_on_plane: numpy array of shape (3,) representing a point
            on the plane.

        Returns:
        - projected_points: numpy array of shape (N, 3) representing the
            projected 3D point cloud.
        """
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        d = -plane_normal.dot(plane_point)
        dot_product = np.dot(point_cloud_3d, plane_normal)
        t = -(dot_product + d) / (plane_normal ** 2).sum()
        projection = point_cloud_3d - t[:, np.newaxis] * plane_normal
        return projection

    def get_top_most_projected_points(self, projected_point_cloud_3d, filter_percent=0):
        x_vals = projected_point_cloud_3d[:, 0]
        z_vals = projected_point_cloud_3d[:, 2]
        points_by_x_val = {}
        for i, x in enumerate(x_vals):
            z = z_vals[i]
            if x in points_by_x_val:
                points_by_x_val[x].add(z)
            else:
                points_by_x_val[x] = set([z])
        x_keys = np.fromiter(points_by_x_val.keys(), dtype=float)
        top_points = np.array([x_keys, np.zeros_like(x_keys)]).T
        for i, key in enumerate(x_keys):
            z_vals = points_by_x_val[key]
            top_points[i, 1] = max(list(z_vals))
        top_points = top_points[top_points[:, 0].argsort()]
        left_index = int(len(x_keys) * filter_percent / 200)
        return top_points[left_index:-left_index]

    def get_implant_x_plane(self, top_most_projected_points):
        x, z = top_most_projected_points.T
        dz = np.abs(np.gradient(z))
        return x[np.argmax(dz)]

    def __get_implant_roi_point_coloud(self, bone_mesh, filter_percent=25):
        medial_point_cloud = self.medial_point_clouds[bone_mesh.name]
        projected_point_cloud_3d = self.project_3d_to_plane(medial_point_cloud[:, :3],
                                                            np.array([0, 1, 0]))
        top_most_projected_points = self.get_top_most_projected_points(
            projected_point_cloud_3d,
            filter_percent)
        implant_x_plane = self.get_implant_x_plane(top_most_projected_points)
        tibia_side = self.sides[bone_mesh.name]
        if tibia_side == 'left':
            return medial_point_cloud[medial_point_cloud[:, 0] < implant_x_plane]
        return medial_point_cloud[medial_point_cloud[:, 0] > implant_x_plane]
