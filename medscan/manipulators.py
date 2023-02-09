import numpy as np


class PointCloudManipulator:
    def __init__(self, medial_point_cloud):
        self.roi_points = medial_point_cloud
        centered_points = self.__get_centered_point_cloud(
            self.roi_points)
        self.centred_nomalised_points = self.__get_normalized_point_cloud(
            centered_points)

    def __get_normalized_point_cloud(self, point_cloud):
        x, y, z, density = zip(*point_cloud)
        density_min, density_max = min(density), max(density)
        normalized_density = [(d - density_min) /
                              (density_max - density_min) for d in density]
        return np.array(list(zip(x, y, z, normalized_density)))

    def __get_centered_point_cloud(self, point_cloud):
        """
        Centers a 4D point cloud (x, y, z, density) by subtracting the mean of the x, y, z coordinates.

        Args:
            point_cloud (np.array): 4D array of shape (n, 4) where n is the number of points

        Returns:
            np.array: Centered 4D point cloud of shape (n, 4)
        """
        mean = np.mean(point_cloud[:, :3], axis=0)
        centered_cloud = point_cloud.copy()
        centered_cloud[:, :3] -= mean
        return centered_cloud
