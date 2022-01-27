from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

n_clusters = 4
n_features = 2
X, y = make_blobs(500, centers=n_clusters, n_features=n_features, cluster_std=0.8)
clf = KMeans(n_clusters=n_clusters, random_state=1).fit(X)
y = clf.fit_predict(X)
centroid = clf.cluster_centers_
means = np.zeros((n_clusters, n_features))
means[:, :] = centroid[:]
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.scatter(means[:, 0], means[:, 1], c='k', marker='x')
plt.show()
