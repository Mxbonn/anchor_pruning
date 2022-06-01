import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


def _kmedoids(data, initial_medoids, delta=.001, maxiter=2, metric="euclidean",
              stochastic=False, stochastic_samples=100):
    medoids = initial_medoids.copy()
    n, d = data.shape
    k, cdim = medoids.shape
    assert d == cdim

    prev_dist = 0
    for i in tqdm(range(0, maxiter)):
        distance_matrix = cdist(data, medoids, metric=metric)
        cluster_ids = distance_matrix.argmin(axis=1)
        distances = distance_matrix[np.arange(len(distance_matrix)), cluster_ids]
        avg_distance = distances.mean()
        if (1 - delta) * prev_dist <= avg_distance <= prev_dist:
            print("Stopped early due to update smaller then delta.")
            break
        prev_dist = avg_distance
        for j in range(k):  # (1 pass in C)
            c = np.where(cluster_ids == j)[0]
            if len(c) > 0:
                medoids[j] = data[c].mean(axis=0)
                if not stochastic:
                    medoids[j] = get_new_medoid(data[c], metric)
                else:
                    data_cluster = data[c][
                        np.random.choice(data[c].shape[0], min(stochastic_samples, len(data[c])), replace=False)]
                    medoids[j] = get_new_medoid(data_cluster, metric)
    return medoids, cluster_ids


def kmedoids(data, initial_medoids, delta=.001, maxiter=2, metric="euclidean"):
    return _kmedoids(data, initial_medoids, delta=delta, maxiter=maxiter, metric=metric)


def kmedoids_stochastic(data, initial_medoids, delta=.001, maxiter=2, metric="euclidean", max_samples=100):
    return _kmedoids(data, initial_medoids, delta=delta, maxiter=maxiter, metric=metric,
                     stochastic=True, stochastic_samples=max_samples)


def get_new_medoid(data_cluster, metric):
    dissimilarity = float("inf")
    new_medoid = None

    for data_point in data_cluster:
        distance_matrix = cdist(data_cluster, np.reshape(data_point, (1, data_point.size)), metric=metric)
        new_dissimilarity = np.sum(distance_matrix)
        if new_dissimilarity < dissimilarity:
            dissimilarity = new_dissimilarity
            new_medoid = data_point

    return new_medoid

