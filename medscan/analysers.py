import numpy as np
# import scipy.stats as stats
from scipy.stats import skew, norm, gaussian_kde, skewnorm
from scipy.optimize import minimize
from .helpers import geometry as geom


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
    def __init__(self, convex_hull):
        self.convex_hull = convex_hull
        self.volume = self.get_convex_hull_volume()

    def get_convex_hull_volume(self):
        simplices = np.column_stack((np.repeat(self.convex_hull.vertices[0], self.convex_hull.nsimplex),
                                     self.convex_hull.simplices))
        tets = self.convex_hull.points[simplices]
        return np.sum(geom.tetrahedron_volume(tets[:, 0], tets[:, 1],
                                              tets[:, 2], tets[:, 3]))
