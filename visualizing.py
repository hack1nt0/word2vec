__author__ = 'DY'

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def visualize(visualizeVecs, texts, header="figure"):
    fig = plt.figure(header)

    if (visualizeVecs.shape[1] > 3):
        covariance = visualizeVecs.T.dot(visualizeVecs)
        U,S,V = np.linalg.svd(covariance)
        coord = (visualizeVecs - np.mean(visualizeVecs, axis=0)).dot(U[:,0:3])
    else:
        coord = visualizeVecs

    ax = fig.add_subplot(111, projection='3d')
    for x, y, z, text in zip(coord[:, 0], coord[:, 1], coord[:, 2], texts):
        ax.text(x, y, z, text)
        ax.scatter(x, y, z)


    ax.set_xlim3d(min(coord[:, 0]), max(coord[:, 0]))
    ax.set_ylim3d(min(coord[:, 1]), max(coord[:, 1]))
    ax.set_zlim3d(min(coord[:, 2]), max(coord[:, 2]))

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.savefig("img/" + header)
    plt.show()

# visualize(np.random.rand(2, 10), ["423424", "3432423"], "hello")
