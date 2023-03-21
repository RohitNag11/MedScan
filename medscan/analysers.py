import numpy as np
# import scipy.stats as stats
from scipy.stats import skew, norm, gaussian_kde, skewnorm
from scipy.optimize import minimize
from .helpers import geometry as geom
from scipy.spatial import ConvexHull


class PointCloudAnalyser:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        self.density = point_cloud[:, 3]
        # self.mean_fit, self.std_fit, self.skew_fit = self.__fit_skewed_normal()

    def get_density_histogram(self, bins=10):
        return np.histogram(self.density, bins=bins)

    def get_density_kde(self):
        data = self.density
        data = data.reshape(-1, 1)
        density_distribution = gaussian_kde(data.T)
        density = np.linspace(min(self.density), max(self.density), 100)
        freq = density_distribution(density)
        return density, freq

    def get_skewed_normal_fit(self):
        params = skewnorm.fit(self.density)
        mean, var, skew, kurt = skewnorm.stats(params, moments='mvsk')
        density, freq = self.get_density_kde()
        skew_norm = skewnorm.pdf(density, *params)
        skew_norm = skew_norm * max(freq) / max(skew_norm)
        return density, skew_norm

    def get_n_percentile(self, percentile=90):
        return np.percentile(self.density, percentile)


class ConvexHullAnalyser:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        self.convex_hull_3d = self.__convex_hull_3d()
        self.volume = self.__get_convex_hull_volume()

    def __convex_hull_3d(self):
        """Compute the convex hull of a 3D point cloud using the gift wrapping algorithm"""
        points = self.point_cloud[:, :3]
        hull = ConvexHull(points)
        return hull

    def sliced_convex_hull_2d(self):
        def convex_hull_2d(point_cloud):
            # Construct a 2D array of x and y coordinates
            xy_coords = point_cloud[:, :2]
            # Compute the convex hull of the 2D points
            hull = ConvexHull(xy_coords)
            vertices = np.take(point_cloud, hull.vertices, axis=0)
            return vertices
        point_cloud = self.point_cloud
        points_by_z = geom.split_point_cloud_by_z(point_cloud)
        point_centers = np.array([np.mean(point_cloud, axis=0)
                                  for point_cloud in points_by_z])
        convex_hull_vertices_by_z = []
        for i, point_cloud in enumerate(points_by_z):
            if len(point_cloud) > 3:
                convex_hull_vertices = convex_hull_2d(point_cloud)
                convex_hull_vertices_by_z.append(convex_hull_vertices)
        hull_centers = np.array([np.mean(hull, axis=0)
                                 for hull in convex_hull_vertices_by_z])
        return convex_hull_vertices_by_z, hull_centers, point_centers

    def __get_convex_hull_volume(self):
        simplices = np.column_stack((np.repeat(self.convex_hull_3d.vertices[0], self.convex_hull_3d.nsimplex),
                                     self.convex_hull_3d.simplices))
        tets = self.convex_hull_3d.points[simplices]
        return np.sum(geom.tetrahedron_volume(tets[:, 0], tets[:, 1],
                                              tets[:, 2], tets[:, 3]))
