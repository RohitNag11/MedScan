# from matplotlib import cm
# from silx.math.colormap import cmap
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from medscan import (readers as msr,
                     viewers as msv,
                     segmenters as mss)
import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy import (unique, where)
import sklearn
from sklearn.cluster import (Birch,
                             KMeans,
                             MiniBatchKMeans,
                             SpectralClustering)
from collections import Counter


def model_predict(model, X):
    # fit the model
    model.fit(X)
    y_pred = []
    try:
        y_pred = model.predict(X)
    except:
        y_pred = model.fit_predict(X)
        print("No 'predict' method in model")
    return y_pred


def plot_clusters(ax, y_pred, X, title='Model X', colors=['b', 'g', 'r']):
    y_pred_counts = np.array(Counter(y_pred).most_common())
    # ordered_colors = [colors[class_count[0]] for class_count in y_pred_counts]

    # retrieve unique clusters
    clusters = y_pred_counts[:, 0]
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # create scatter plot for samples from each cluster
    for i, cluster in enumerate(clusters):
        # get row indexes for samples with this cluster
        row_ix = where(y_pred == cluster)
        # create scatter of these samples
        ax.scatter(X[row_ix, 0], X[row_ix, 1],
                   X[row_ix, 2], alpha=0.5, ec=None, c=colors[i])


with open('lt_pointcloud.pickle', 'rb') as data:
    X = pickle.load(data)

X = X[X[:, 2] > 80]
X = X[X[:, 0] < np.mean(X[:, 0])]


lt_bone_mesh = msr.BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl', 'Left Tibia')

down_sampled_plot = msv.Density4DSliderPlot(X,
                                            lt_bone_mesh,
                                            0,
                                            300)


class BandThresh:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X):
        d_bands = list(np.linspace(
            min(X[:, 3]) - 1, max(X[:, 3]) + 1, self.n_clusters + 1))
        self.d_bands = d_bands[::-1]

    def predict(self, X):
        d_array = X[:, 3]
        y = [self.__get_band(d, self.d_bands) for d in d_array]
        return y

    def __get_band(self, d, d_bands):
        for i, thres in enumerate(d_bands):
            if d >= thres:
                return i - 1
        print('failed')


# models and predictions
n_clusters = 3

model_bandthresh = BandThresh(n_clusters)
y_pred_bandthresh = model_predict(model_bandthresh, X)

model_birch = Birch(threshold=0.01, n_clusters=n_clusters)
y_pred_birch = model_predict(model_birch, X)

model_kmeans = KMeans(n_clusters=n_clusters)
y_pred_kmeans = model_predict(model_kmeans, X)

model_minibatch = MiniBatchKMeans(n_clusters=n_clusters)
y_pred_minibatch = model_predict(model_minibatch, X)

model_spectralclust = SpectralClustering(n_clusters=n_clusters)
y_pred_spectralclust = model_predict(model_spectralclust, X)


fig = plt.figure()
fig.suptitle('Left Tibia Predected Clusters')

ax0 = fig.add_subplot(231, projection='3d')
plot_clusters(ax0, y_pred_bandthresh, X, 'Band-Thresh')

ax1 = fig.add_subplot(232, projection='3d')
plot_clusters(ax1, y_pred_birch, X, 'BIRCH')

ax2 = fig.add_subplot(233, projection='3d')
plot_clusters(ax2, y_pred_kmeans, X, 'K-Means')

ax3 = fig.add_subplot(234, projection='3d')
plot_clusters(ax3, y_pred_minibatch, X, 'Mini-Batch K-Means')

ax4 = fig.add_subplot(235, projection='3d')
plot_clusters(ax4, y_pred_spectralclust, X, 'Spectral Clustering')

# show the plot
plt.show()


# def cuboid_data(o, size=(1, 1, 1)):
#     X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
#          [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
#          [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
#          [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
#          [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
#          [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
#     X = np.array(X).astype(float)
#     for i in range(3):
#         X[:, :, i] *= size[i]
#     X += np.array(o)
#     return X


# def plotCubeAt(positions, sizes=None, colors=None, **kwargs):
#     if not isinstance(colors, (list, np.ndarray)):
#         colors = ["C0"]*len(positions)
#     if not isinstance(sizes, (list, np.ndarray)):
#         sizes = [(4, 4, 4)]*len(positions)
#     g = []
#     for p, s, c in zip(positions, sizes, colors):
#         g.append(cuboid_data(p, size=s))
#     return Poly3DCollection(np.concatenate(g),
#                             facecolors=np.repeat(colors, 6, axis=0), **kwargs)


# positions = X[:, :3]
# normalised_density = (X[:, 3] - np.min(X[:, 3]))/np.ptp(X[:, 3])
# colors = cm.jet(normalised_density)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# pc = plotCubeAt(positions, colors=colors, edgecolor=None, alpha=0.1)
# ax.add_collection3d(pc)

# ax.set_xlim([min(X[:, 0]), max(X[:, 0])])
# ax.set_ylim([min(X[:, 1]), max(X[:, 1])])
# ax.set_zlim([min(X[:, 2]), max(X[:, 2])])

# ax.set_aspect('equal')

# plt.show()
