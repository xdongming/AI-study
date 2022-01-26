from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

'''
网格点处理
'''
x, y = make_blobs(n_samples=50, centers=2, cluster_std=0.6)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='rainbow')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
axisx = np.linspace(xlim[0], xlim[1], 30)
axisy = np.linspace(ylim[0], ylim[1], 30)
axisx, axisy = np.meshgrid(axisx, axisy)
xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
'''
训练与绘图
'''
clf = SVC(kernel='linear').fit(x, y)
z = SVC.decision_function(clf, xy).reshape(axisx.shape)
ax.contour(axisx, axisy, z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.show()