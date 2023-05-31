import numpy as np
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans, SpectralClustering, DBSCAN
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.stats import zscore
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from .helpers import geometry as geom
import pandas as pd


class PointCloudClassifier:
    def __init__(self, thresholded_point_cloud):
        # Gets the x, y, z coordinates
        self.X_pre_filter = thresholded_point_cloud
        self.X_filtered_1, self.filter_1_labels = self.__birch_filter(self.X_pre_filter)
        self.X_filtered_2, self.filter_2_labels = self.__dbscan_filter(
            self.X_filtered_1, eps_fraction=0.2, min_samples=7
        )

    def __model_predict(self, points, model):
        """Fits the model to the data and returns the predicted classes for the points in the point cloud."""
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
        """Returns the points in the point cloud that belong to the specified class."""
        birch_model = Birch(threshold=threshold, n_clusters=n_clusters)
        y_pred = self.__model_predict(points, birch_model)
        mean_y_pos_per_cluster = [
            np.mean(points[y_pred == i, 1]) for i in range(n_clusters)
        ]
        return points[y_pred == np.argmin(mean_y_pos_per_cluster)], y_pred

    def __dbscan_filter(self, point_cloud, eps_fraction=0.2, min_samples=2):
        """Returns the points in the point cloud that belong to the specified class."""
        points = point_cloud[:, :3]
        eps = points[:, 1].std() * eps_fraction
        db_model = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db_model.labels_
        sorted_labels = self.__sort_labels(labels)
        return point_cloud[sorted_labels == 0], sorted_labels

    def cluster_points(self, point_cloud, threshold):
        # calculate pairwise distances between all points in the point cloud
        distances = np.sum((point_cloud[:, np.newaxis, :] - point_cloud) ** 2, axis=-1)
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
                closest_indices = np.argwhere(distances[i] < threshold).flatten()
                # assign the same cluster label to all closest points
                labels[closest_indices] = num_clusters
        # return the clustered point cloud and the cluster labels
        sorted_labels = self.__sort_labels(labels)
        return sorted_labels

    def __sort_labels(self, labels):
        """Returns the labels in the order of their frequency."""
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
        label_mapping = {
            label: i if label != -1 else -1 for i, label in enumerate(sorted_labels)
        }
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
        point_centers = np.array(
            [np.mean(point_cloud, axis=0) for point_cloud in points_by_z]
        )
        convex_hull_vertices_by_z = []
        for i, point_cloud in enumerate(points_by_z):
            if len(point_cloud) > 3:
                convex_hull_vertices = convex_hull_2d(point_cloud)
                convex_hull_vertices_by_z.append(convex_hull_vertices)
        hull_centers = np.array(
            [np.mean(hull, axis=0) for hull in convex_hull_vertices_by_z]
        )
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

    def convex_hull_3d(self, point_cloud):
        """Compute the convex hull of a 3D point cloud using the gift wrapping algorithm"""
        points = point_cloud[:, :3]
        hull = ConvexHull(points)
        return hull


class ImplantPegResultsClassifier:
    def __init__(self, res_path):
        self.df = pd.read_csv(res_path)
        self.columns = self.df.columns

    def __name_param(self, param, append_mm=True):
        # split on underscores and capitalise first letter of each word
        name = " ".join([word.capitalize() for word in param.split("_")])
        if append_mm:
            # Add (mm) to any parameter that is not a density
            if "density" not in name:
                name += " (mm)"
        return name

    def __to_ordinal(self, n: int):
        if 11 <= (n % 100) <= 13:
            suffix = "th"
        else:
            suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
        return str(n) + suffix

    def __regression_2d(self, input_array, output_array, deg):
        # Check if the length of input_array and output_array are the same
        if len(input_array) != len(output_array):
            raise ValueError("input_array and output_array must have the same length")
        # Reshape input array
        X = np.array(input_array).reshape(-1, 1)
        y = np.array(output_array)
        # Identify outliers
        z_scores = np.abs(zscore(y))
        outliers = z_scores >= 3
        outlier_values = y[outliers]
        # Remove outliers
        filtered_entries = z_scores < 3
        X = X[filtered_entries]
        y = y[filtered_entries]
        # Linear regression
        if deg == 1:
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            coeff = model.coef_
            intercept = model.intercept_
        # Polynomial regression
        else:
            poly = PolynomialFeatures(degree=deg)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)
            y_pred = model.predict(X_poly)
            coeff = model.coef_
            intercept = model.intercept_
        # R2 score
        r2 = r2_score(y, y_pred)
        # Residuals
        residuals = y - y_pred
        return {
            "intercept": intercept,
            "coefficients": coeff,
            "r2_score": r2,
            "residuals": residuals,
            "outliers": outlier_values,
        }

    def analyse_data_2d(self, deg, input_columns, output_columns):
        if output_columns is None:
            output_columns = [col for col in self.columns if col not in input_columns]
        for input_col in input_columns:
            if input_col not in self.columns:
                raise ValueError(f"Column {input_col} does not exist in the data.")
            ncols = np.ceil(len(output_columns) ** 0.5).astype(int)
            nrows = np.ceil(len(output_columns) / ncols).astype(int)
            size_label = self.__name_param(input_col)
            fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharex=True)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle(
                f"{size_label} vs. Other Factors ({self.__to_ordinal(deg)} Degree Reg. Fits)",
                fontsize=10,
            )
            for i, output_col in enumerate(output_columns):
                factor_label = self.__name_param(output_col)
                ax_i, ax_j = i // ncols, i % ncols
                input_array = self.df[input_col].to_numpy()
                output_array = self.df[output_col].to_numpy()
                regression_result = self.__regression_2d(input_array, output_array, deg)
                # Separate outliers
                is_outlier = np.isin(output_array, regression_result["outliers"])
                outliers_x = input_array[is_outlier]
                outliers_y = output_array[is_outlier]
                # Non-outliers
                regular_x = input_array[~is_outlier]
                regular_y = output_array[~is_outlier]

                # Plot regular data and outliers
                ax[ax_i, ax_j].scatter(regular_x, regular_y, color="b")
                ax[ax_i, ax_j].scatter(outliers_x, outliers_y, color="r")

                # Plot fit
                sorted_input = np.linspace(regular_x.min(), regular_x.max(), 500)
                poly = PolynomialFeatures(degree=deg)
                sorted_input_transformed = poly.fit_transform(
                    sorted_input.reshape(-1, 1)
                )
                sorted_output = regression_result[
                    "intercept"
                ] + sorted_input_transformed.dot(regression_result["coefficients"])
                ax[ax_i, ax_j].plot(sorted_input, sorted_output, color="r", lw=0.5)

                # Calculate fitted values for residuals
                regular_input_transformed = poly.fit_transform(regular_x.reshape(-1, 1))
                regular_y_hat = regression_result[
                    "intercept"
                ] + regular_input_transformed.dot(regression_result["coefficients"])

                # Plot residuals
                for x, y, y_hat in zip(regular_x, regular_y, regular_y_hat):
                    ax[ax_i, ax_j].plot(
                        [x, x], [y, y_hat], color="gray", linestyle="dotted"
                    )

                ax[ax_i, ax_j].set_title(
                    f"vs. {self.__name_param(output_col, append_mm=False)} (R2 = {regression_result['r2_score']:.2f})",
                    fontsize=8,
                )
                ax[ax_i, ax_j].set_xlabel(f"{size_label}", fontsize=6)
                ax[ax_i, ax_j].set_ylabel(f"{factor_label}", fontsize=6)
            plt.show()

    def __regression_3d(self, X, y, deg, outlier_thresh):
        # Create polynomial features
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(X)
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_poly, y)
        # Make predictions
        y_pred = model.predict(X_poly)
        # Compute residuals
        residuals = y - y_pred
        # Find outliers
        residuals = np.array(residuals)
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (outlier_thresh * iqr)
        upper_bound = q3 + (outlier_thresh * iqr)
        outliers = (residuals < lower_bound) | (residuals > upper_bound)
        # Compute r2 score
        r2 = r2_score(y, y_pred)
        return model, poly, y_pred, residuals, outliers, r2

    def analyse_data_3d(self, deg, input_columns, output_columns, outlier_thresh):
        X = self.df[input_columns]
        if output_columns is None:
            output_columns = [col for col in self.columns if col not in input_columns]
        for output_col in output_columns:
            y = self.df[output_col]
            model, poly, y_pred, residuals, outliers, r2 = self.__regression_3d(
                X, y, deg, outlier_thresh
            )
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
            # Plot surface
            x1_values = np.linspace(
                X[input_columns[0]].min(), X[input_columns[0]].max(), num=25
            )
            x2_values = np.linspace(
                X[input_columns[1]].min(), X[input_columns[1]].max(), num=25
            )
            x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
            X_grid = pd.DataFrame(
                {input_columns[0]: x1_grid.ravel(), input_columns[1]: x2_grid.ravel()}
            )
            X_grid_poly = poly.transform(X_grid)
            y_grid_pred = model.predict(X_grid_poly)
            y_grid_pred = y_grid_pred.reshape(x1_grid.shape)
            ax.plot_surface(
                x1_grid,
                x2_grid,
                y_grid_pred,
                cmap="viridis",
                alpha=0.4,
            )
            # Plot actual values
            ax.scatter(
                X[~outliers][input_columns[0]],
                X[~outliers][input_columns[1]],
                y[~outliers],
                color="blue",
                label="Inliers",
            )
            # Plot outliers
            ax.scatter(
                X[outliers][input_columns[0]],
                X[outliers][input_columns[1]],
                y[outliers],
                color="red",
                label="Outliers",
            )
            # Plot predicted values
            ax.scatter(
                X[input_columns[0]],
                X[input_columns[1]],
                y_pred,
                color="green",
                label="Predicted values",
            )
            # Plot residuals as dotted straight lines
            for i in range(X.shape[0]):
                ax.plot(
                    [X[input_columns[0]].iloc[i], X[input_columns[0]].iloc[i]],
                    [X[input_columns[1]].iloc[i], X[input_columns[1]].iloc[i]],
                    [y.iloc[i], y_pred[i]],
                    color="gray",
                    linestyle="dotted",
                )
            ax.legend()
            ax.set_title(
                f"Implant Size vs {self.__name_param(output_col, append_mm=False)}, {self.__to_ordinal(deg)}-order Poly Fit (R2 = {r2:.2f})"
            )
            ax.set_xlabel(self.__name_param(input_columns[0]))
            ax.set_ylabel(self.__name_param(input_columns[1]))
            ax.set_zlabel(self.__name_param(output_col))

            plt.show()

    def __regression_multi_param(self, X, Y, deg):
        # Create polynomial features
        poly = PolynomialFeatures(deg)
        X_poly = poly.fit_transform(X)

        # Fit multi output regression model
        model = MultiOutputRegressor(LinearRegression())
        model.fit(X_poly, Y)

        # Make predictions
        Y_pred = model.predict(X_poly)

        coef_matrix = []
        for i, output_col in enumerate(Y.columns):
            coefs = model.estimators_[i].coef_
            intercept = model.estimators_[i].intercept_
            coef_vector = [intercept] + list(coefs)
            coef_matrix.append(coef_vector)
        return model, poly, Y_pred, coef_matrix

    def analyse_data_multi_param(self, deg, input_columns, output_columns):
        if output_columns is None:
            output_columns = [col for col in self.columns if col not in input_columns]
        X = self.df[input_columns]
        Y = self.df[output_columns]

        model, poly, Y_pred, coef_matrix = self.__regression_multi_param(X, Y, deg)

        for i, output_col in enumerate(output_columns):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")

            # Plot actual values
            ax.scatter(
                X[input_columns[0]],
                X[input_columns[1]],
                Y[output_col],
                color="blue",
                label="Actual values",
            )

            # Plot predicted values
            ax.scatter(
                X[input_columns[0]],
                X[input_columns[1]],
                Y_pred[:, i],
                color="green",
                label="Predicted values",
            )

            # Plot residuals
            residuals = Y[output_col] - Y_pred[:, i]
            ax.quiver(
                X[input_columns[0]],
                X[input_columns[1]],
                Y_pred[:, i],
                0,
                0,
                residuals,
                color="grey",
                arrow_length_ratio=0,
                label="Residuals",
            )

            # Plot surface
            x1_values = np.linspace(
                X[input_columns[0]].min(), X[input_columns[0]].max(), num=20
            )
            x2_values = np.linspace(
                X[input_columns[1]].min(), X[input_columns[1]].max(), num=20
            )
            x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
            X_grid = pd.DataFrame(
                {input_columns[0]: x1_grid.ravel(), input_columns[1]: x2_grid.ravel()}
            )
            X_grid_poly = poly.transform(X_grid)
            y_grid_pred = model.predict(X_grid_poly)[:, i]
            y_grid_pred = y_grid_pred.reshape(x1_grid.shape)
            ax.plot_surface(
                x1_grid,
                x2_grid,
                y_grid_pred,
                cmap="viridis",
                alpha=0.4,
            )

            ax.legend()
            ax.set_title(output_col)
            ax.set_xlabel(input_columns[0])
            ax.set_ylabel(input_columns[1])
            ax.set_zlabel(output_col)

            plt.show()

    def generate_X_fit(self, X, num_points=100):
        # For each column in X, generate a linspace between the min and max value of the column
        X_values = [
            np.linspace(X[col].min(), X[col].max(), num_points) for col in X.columns
        ]

        # Generate a meshgrid for the X values
        mesh = np.meshgrid(*X_values)

        # Reshape the meshgrids to create columns of the input data frame
        X_fit_values = [axis.ravel() for axis in mesh]

        # Create a DataFrame for X_fit
        X_fit = pd.DataFrame(
            {col: values for col, values in zip(X.columns, X_fit_values)}
        )

        return X_fit

    def get_output_from_inputs(self, deg, input_dict, output_columns):
        input_columns = list(input_dict.keys())
        if output_columns is None:
            output_columns = [col for col in self.columns if col not in input_columns]
        X = self.df[input_columns]
        Y = self.df[output_columns]

        model, poly, Y_pred, coef_matrix = self.__regression_multi_param(X, Y, deg)
        # input_values = np.array(input_values).reshape(1, -1)
        X_fit = pd.DataFrame(input_dict)
        Y_fit = model.predict(poly.transform(X_fit))

        output = {col: Y_fit[:, i] for i, col in enumerate(output_columns)}
        return output

    def __batch_inputs(self, deg, X, Y, batch_size):
        model, poly, Y_pred, coef_matrix = self.__regression_multi_param(X, Y, deg)
        X_fit = self.generate_X_fit(X, num_points=1000)
        Y_fit = model.predict(poly.transform(X_fit))
        # Use KMeans to cluster the predicted outputs into n groups
        kmeans = KMeans(n_clusters=batch_size, random_state=0).fit(Y_fit)

        # The labels_ attribute contains the cluster assignment for each input
        batches = kmeans.labels_

        return batches, X_fit, Y_fit

    def analyse_batch_locations(
        self, deg, input_columns, output_columns, batch_size, output_path=None
    ):
        if output_columns is None:
            output_columns = [col for col in self.columns if col not in input_columns]

        X = self.df[input_columns]
        Y = self.df[output_columns]

        batches, X_fit, Y_fit = self.__batch_inputs(deg, X, Y, batch_size)

        fig, ax = plt.subplots(figsize=(12, 8))

        batch_values = np.zeros((batch_size, X.shape[1] + Y.shape[1]))

        # For each unique batch
        for i, batch in enumerate(np.unique(batches)):
            # Select points in this batch
            X_batch = X_fit[batches == batch]
            # Select corresponding predictions
            Y_fit_batch = Y_fit[batches == batch]
            # Calculate the average prediction for this batch
            Y_pred_avg = np.mean(Y_fit_batch, axis=0)
            # Plot points in this batch
            ax.scatter(
                X_batch[X.columns[0]], X_batch[X.columns[1]], label=f"Batch {batch}"
            )

            # Add text label for average x[0] and x[1]:
            batch_x0_avg = np.mean(X_batch[X.columns[0]])
            batch_x1_avg = np.mean(X_batch[X.columns[1]])
            batch_values[i] = np.array([batch_x0_avg, batch_x1_avg, *Y_pred_avg])
            ax.text(
                batch_x0_avg,
                batch_x1_avg - 0.2,
                f"x̄={batch_x0_avg:.2f}, ȳ={batch_x1_avg: .2f}",
                {
                    "color": "black",
                    "fontsize": 10,
                    "ha": "center",
                    "va": "top",
                    "bbox": dict(
                        boxstyle="round", fc="#FFFFFFBD", ec="#00000080", pad=0.2
                    ),
                },
            )
            # Add a cross at the average position for each batch
            ax.scatter(batch_x0_avg, batch_x1_avg, marker="x", color="black")
        ax.set_xlabel(self.__name_param(input_columns[0]))
        ax.set_ylabel(self.__name_param(input_columns[1]))
        ax.set_title(
            f"Batch locations using KMeans clustering on predicted outputs using {self.__to_ordinal(deg)}-order polynomial"
        )
        # Show the plot with a legend
        plt.legend()
        plt.show()
        batch_values_pd = batch_values = pd.DataFrame(
            batch_values, columns=[*X.columns, *Y.columns]
        )
        if output_path is not None:
            batch_values_pd.to_csv(output_path)

    def visualise_output_for_batches(
        self, deg, input_columns, batch_size, output_columns=None
    ):
        if output_columns is None:
            output_columns = [col for col in self.columns if col not in input_columns]

        X = self.df[input_columns]
        Y = self.df[output_columns]

        batches, X_fit, Y_fit = self.__batch_inputs(deg, X, Y, batch_size)

        unique_batches = np.unique(batches)

        # For each unique batch
        for batch_idx, batch in enumerate(unique_batches):
            # Select points in this batch
            X_batch = X_fit[batches == batch]

            # Select corresponding predictions
            y_fit_batch = Y_fit[batches == batch]

            # Calculate the average prediction for this batch
            y_fit_batch_avg = np.mean(y_fit_batch, axis=0)

            # Calculate the number of rows and columns for the grid
            num_cols = int(np.ceil(np.sqrt(len(output_columns))))
            num_rows = int(np.ceil(len(output_columns) / num_cols))

            # Create a new figure with a grid of subplots
            fig, axs = plt.subplots(
                num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows)
            )
            axs = axs.ravel()  # Flatten the array of axes, for easier indexing

            # Create hexbin plot for each output column in this batch
            for col_idx, col in enumerate(Y.columns):
                hb = axs[col_idx].hexbin(
                    X_batch[X.columns[0]],
                    X_batch[X.columns[1]],
                    C=y_fit_batch[:, col_idx],
                    gridsize=50,
                    cmap="viridis",
                )
                fig.colorbar(hb, ax=axs[col_idx]).set_label(
                    self.__name_param(col), fontsize=8
                )
                axs[col_idx].set_title(
                    f"Batch {batch}, Avg {self.__name_param(col, append_mm=False)}: {y_fit_batch_avg[col_idx]:.2f} mm",
                    fontsize=10,
                )
                axs[col_idx].set_xlabel(self.__name_param(input_columns[0]), fontsize=8)
                axs[col_idx].set_ylabel(self.__name_param(input_columns[1]), fontsize=8)
                axs[col_idx].tick_params(axis="both", which="major", labelsize=8)
                axs[col_idx].tick_params(axis="both", which="minor", labelsize=8)

            # Remove empty subplots (if the number of output columns is not a perfect square)
            for col_idx in range(len(Y.columns), num_rows * num_cols):
                fig.delaxes(axs[col_idx])

            # Adjust the layout to prevent overlap between subplots
            fig.tight_layout()
            plt.show()

            # Print predictions for this batch
            print(f"Predictions for batch {batch}:")
            print(y_fit_batch_avg)
