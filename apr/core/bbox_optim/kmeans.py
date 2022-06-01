import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


def kmeans(data, initial_centers, delta=.001, maxiter=2, metric="euclidean"):
    centers = initial_centers.copy()
    n, d = data.shape
    k, cdim = centers.shape
    assert d == cdim

    prev_dist = 0
    for i in tqdm(range(0, maxiter)):
        distance_matrix = cdist(data, centers, metric=metric)
        nearest_centers = distance_matrix.argmin(axis=1)
        distances = distance_matrix[np.arange(len(distance_matrix)), nearest_centers]
        avg_distance = distances.mean()
        if (1 - delta) * prev_dist <= avg_distance <= prev_dist:
            print("Stopped early due to update smaller then delta.")
            break
        prev_dist = avg_distance
        for j in range(k):  # (1 pass in C)
            c = np.where(nearest_centers == j)[0]
            if len(c) > 0:
                centers[j] = data[c].mean(axis=0)
    return centers, nearest_centers
