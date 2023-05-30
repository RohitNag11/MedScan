import numpy as np
from .helpers import geometry as geom


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
        centered_points = self.__get_point_cloud_in_tibia_space(self.roi_points, side)
        self.centred_nomalised_points = self.__get_normalized_point_cloud(
            centered_points
        )
        if side == "left":
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

    def __get_point_cloud_in_tibia_space(self, point_cloud, side):
        """
        Dimensionally normalises the point cloud to the tibial coordinate system.
        - If the 'side' argument is 'left', it mirrors the point cloud across the y-axis.
        - The origin is set to the top slices left most point in the x direction.

        Args:
        - point_cloud: a numpy array of shape (N, 4) representing the point cloud
        - side: a string indicating which side the point cloud is from ('left' or 'right')

        Returns:
        - a numpy array of shape (N, 4) representing the normalized point cloud
        """
        point_cloud_copy = point_cloud.copy()
        if side == "right":
            point_cloud_copy[:, 0] *= -1
        # Make the x-axis origin the left most point in the x direction
        origin_x = np.min(point_cloud_copy[:, 0])
        # Make the z-axis origin the top most point in the z direction
        origin_z = np.max(point_cloud_copy[:, 2])
        # Find all points that have the same x coordinate as the origin
        origin_x_points = point_cloud_copy[point_cloud_copy[:, 0] == origin_x]
        # Make the y-axis origin the mean of the y coordinates of the points with the same x coordinate as the origin
        origin_y = np.mean(origin_x_points[:, 1])
        # Translate the point cloud so that the origin is at [0, 0, 0]:
        point_cloud_copy = point_cloud_copy - np.array(
            [origin_x, origin_y, origin_z, 0]
        )
        return point_cloud_copy

    def convert_point_cloud_to_cn_density_space(self, point_cloud):
        x, y, z, density = zip(*point_cloud)
        cn_density = [
            geom.map_to_range(d, self.original_density_bounds, self.cn_density_bounds)
            for d in density
        ]
        return np.array(list(zip(x, y, z, cn_density)))
