import numpy as np
from sklearn.cluster import (Birch,
                             KMeans,
                             MiniBatchKMeans,
                             SpectralClustering,
                             DBSCAN)
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import trimesh


class PointCloudClassifier:
    def __init__(self, thresholded_point_cloud):
        # Gets the x, y, z coordinates
        self.X_pre_filter = thresholded_point_cloud
        self.X_filtered_1, self.filter_1_labels = self.__birch_filter(
            self.X_pre_filter)
        self.X_filtered_2, self.filter_2_labels = self.__dbscan_filter(
            self.X_filtered_1, 1, 7)

    def __model_predict(self, points, model):
        '''Fits the model to the data and returns the predicted classes for the points in the point cloud.'''
        X = points
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        model.fit(X)
        y_pred = []
        try:
            y_pred = model.predict(X)
        except:
            y_pred = model.fit_predict(X)
            print("No 'predict' method in model")
        return self.__sort_labels(y_pred)

    def __birch_filter(self, points, n_clusters=2, threshold=0.01):
        '''Returns the points in the point cloud that belong to the specified class.'''
        birch_model = Birch(threshold=threshold, n_clusters=n_clusters)
        y_pred = self.__model_predict(points, birch_model)
        return points[y_pred == 0], y_pred

    def __dbscan_filter(self, point_cloud, eps=0.01, min_samples=2):
        '''Returns the points in the point cloud that belong to the specified class.'''
        points = point_cloud[:, :3]
        db_model = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db_model.labels_
        sorted_labels = self.__sort_labels(labels)
        return point_cloud[sorted_labels == 0], sorted_labels

    def __closeness_filter(self, points, threshold=0.01):
        '''Returns the points in the point cloud that are close to each other.'''
        distances = np.sum((points[:, np.newaxis, :] - points)**2, axis=-1)
        # set the diagonal to infinity to ignore the distance from a point to itself
        np.fill_diagonal(distances, np.inf)
        # find the indices of the closest points for each point
        closest_indices = np.argmin(distances, axis=1)
        # find the indices of the points that are close to another point
        mask = np.min(distances, axis=1) < threshold
        # return the filtered point cloud
        return points[mask]

    def cluster_points(self, point_cloud, threshold):
        # calculate pairwise distances between all points in the point cloud
        distances = np.sum(
            (point_cloud[:, np.newaxis, :] - point_cloud)**2, axis=-1)
        # set the diagonal to infinity to ignore the distance from a point to itself
        np.fill_diagonal(distances, np.inf)
        # initialize an array to store the cluster labels
        labels = np.zeros(point_cloud.shape[0], dtype=np.int)
        # initialize a counter for the number of clusters
        num_clusters = 0
        # loop over all points in the point cloud
        for i in range(point_cloud.shape[0]):
            # if the point has not been assigned a cluster label
            if labels[i] == 0:
                # assign a new cluster label to the point
                num_clusters += 1
                labels[i] = num_clusters
                # find the indices of the closest points to the current point
                closest_indices = np.argwhere(
                    distances[i] < threshold).flatten()
                # assign the same cluster label to all closest points
                labels[closest_indices] = num_clusters
        # return the clustered point cloud and the cluster labels
        sorted_labels = self.__sort_labels(labels)
        return sorted_labels

    def __sort_labels(self, labels):
        '''Returns the labels in the order of their frequency.'''
        # find the unique labels and count the number of occurrences of each label
        unique_labels, counts = np.unique(labels, return_counts=True)
        # separate the label with value -1 if it exists
        neg_one_index = np.where(unique_labels == -1)[0]
        if len(neg_one_index) > 0:
            neg_one_label = unique_labels[neg_one_index]
            neg_one_count = counts[neg_one_index]
            unique_labels = np.delete(unique_labels, neg_one_index)
            counts = np.delete(counts, neg_one_index)
        # sort the remaining unique labels based on the number of occurrences
        sorted_labels = unique_labels[np.argsort(counts)][-1::-1]
        # add the label with value -1 back to the end of the sorted labels
        if len(neg_one_index) > 0:
            sorted_labels = np.append(sorted_labels, neg_one_label)
        # create a dictionary that maps the original labels to the new labels, keeping the value -1 labeled as -1
        label_mapping = {label: i if label != -1 else -
                         1 for i, label in enumerate(sorted_labels)}
        # replace the original labels with the new labels
        sorted_labels = np.array([label_mapping[label] for label in labels])
        return sorted_labels

    def sliced_2d_convex_hull(self):
        def convex_hull_2d(point_cloud):
            # Construct a 2D array of x and y coordinates
            xy_coords = point_cloud[:, :2]
            # Compute the convex hull of the 2D points
            hull = ConvexHull(xy_coords)
            vertices = np.take(point_cloud, hull.vertices, axis=0)
            return vertices
        points_by_z = self.__split_point_cloud_by_z()
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

    def __split_point_cloud_by_z(self):
        point_cloud = self.X_filtered_2[:, :3]
        # Sort the point cloud by z value
        sorted_cloud = point_cloud[point_cloud[:, 2].argsort()]
        # Find the indices where the z value changes
        indices = np.where(np.diff(sorted_cloud[:, 2]) != 0)[0] + 1
        # Split the point cloud into separate arrays based on the indices
        point_clouds = np.split(sorted_cloud, indices)
        # Return the point clouds as a list of NumPy arrays
        return point_clouds

    def convex_hull_3d(self):
        """Compute the convex hull of a 3D point cloud using the gift wrapping algorithm"""
        points = self.X_filtered_2[:, :3]
        hull = ConvexHull(points)
        return hull
