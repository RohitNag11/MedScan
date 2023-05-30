import numpy as np
from .helpers import geometry as geom
import medscan.viewers as msv


class PointCloudManipulator:
    def __init__(self, implant_roi_point_cloud, side):
        self.original_x_bounds = (
            min(implant_roi_point_cloud[:, 0]),
            max(implant_roi_point_cloud[:, 0]),
        )
        self.original_y_bounds = (
            min(implant_roi_point_cloud[:, 1]),
            max(implant_roi_point_cloud[:, 1]),
        )
        self.original_z_bounds = (
            min(implant_roi_point_cloud[:, 2]),
            max(implant_roi_point_cloud[:, 2]),
        )
        self.original_density_bounds = (
            min(implant_roi_point_cloud[:, 3]),
            max(implant_roi_point_cloud[:, 3]),
        )
        self.roi_points = implant_roi_point_cloud
        # msv.PointCloudPlot(implant_roi_point_cloud, normalised=False)
        centered_points = self.__get_dim_normalised_point_cloud(self.roi_points, side)
        self.centred_nomalised_points = self.__get_normalized_point_cloud(
            centered_points
        )
        if side == "right":
            self.cn_x_bounds = (
                min(self.centred_nomalised_points[:, 0]),
                max(self.centred_nomalised_points[:, 0]),
            )
        else:
            self.cn_x_bounds = (
                max(self.centred_nomalised_points[:, 0]),
                min(self.centred_nomalised_points[:, 0]),
            )
        self.cn_y_bounds = (
            min(self.centred_nomalised_points[:, 1]),
            max(self.centred_nomalised_points[:, 1]),
        )
        self.cn_z_bounds = (
            min(self.centred_nomalised_points[:, 2]),
            max(self.centred_nomalised_points[:, 2]),
        )
        self.cn_density_bounds = (
            min(self.centred_nomalised_points[:, 3]),
            max(self.centred_nomalised_points[:, 3]),
        )

        self.original_space_bounds = np.array(
            [self.original_x_bounds, self.original_y_bounds, self.original_z_bounds]
        )
        self.cn_space_bounds = np.array(
            [self.cn_x_bounds, self.cn_y_bounds, self.cn_z_bounds]
        )

    def __get_normalized_point_cloud(self, point_cloud):
        x, y, z, density = zip(*point_cloud)
        density_min, density_max = min(density), max(density)
        normalized_density = [
            (d - density_min) / (density_max - density_min) for d in density
        ]
        return np.array(list(zip(x, y, z, normalized_density)))

    def __get_dim_normalised_point_cloud(self, point_cloud, side):
        """
        Normalizes a 4D point cloud to be within a bounding cube of dimension 1 in the first 3 dimensions.
        If the 'side' argument is 'left', it scales the x-values to be between -0.5 and 0.5.
        If the 'side' argument is 'right', it scales the x-values to be between 0.5 and -0.5.
        The y- and z-values are scaled to be between -0.5 and 0.5 in both cases.
        The 4th dimension is left unchanged.

        Args:
        - point_cloud: a numpy array of shape (N, 4) representing the point cloud
        - side: a string indicating which side the point cloud is from ('left' or 'right')

        Returns:
        - a numpy array of shape (N, 4) representing the normalized point cloud
        """
        # Extract the first 3 dimensions of the point cloud
        point_cloud_xyz = point_cloud[:, :3]

        # Compute the bounding box of the point cloud
        min_vals = np.min(point_cloud_xyz, axis=0)
        max_vals = np.max(point_cloud_xyz, axis=0)

        # Compute the center of the bounding box
        center = (min_vals + max_vals) / 2

        # Translate the point cloud to be centered at the origin
        point_cloud_xyz = point_cloud_xyz - center

        # Scale the point cloud to be within a bounding cube of dimension 1
        max_range = np.max(np.abs(point_cloud_xyz[:, 2]))
        scale_factor = 0.5 / max_range
        point_cloud_xyz = point_cloud_xyz * scale_factor

        if side == "left":
            point_cloud_xyz[:, 0] *= -1

        # Concatenate the normalized xyz coordinates with the unchanged 4th dimension
        point_cloud_norm = np.concatenate([point_cloud_xyz, point_cloud[:, 3:]], axis=1)
        return point_cloud_norm
        # point_cloud_norm = np.concatenate(
        #     [point_cloud_xyz, point_cloud[:, 3:]], axis=1)
        # return point_cloud_norm

    def convert_point_cloud_to_cn_density_space(self, point_cloud):
        x, y, z, density = zip(*point_cloud)
        cn_density = [
            geom.map_to_range(d, self.original_density_bounds, self.cn_density_bounds)
            for d in density
        ]
        return np.array(list(zip(x, y, z, cn_density)))
