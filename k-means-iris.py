from collections import defaultdict
import numpy as np
import random
from pprint import pprint
from typing import Iterable
from math import fsum, sqrt
from sklearn import datasets
from functools import partial

Point = np.ndarray
Centroid = Point


def mean(data: Iterable[float]) -> float:
    """Accurate mean of an iterable of floats.

    Args:
        data (Iterable[float]): Iterable of floats. Usually np.ndarray.

    Returns:
        float: Accurate mean of data.
    """
    data = np.array(data)
    return fsum(data) / len(list(data))


def dist(p, q, fsum=fsum, sqrt=sqrt, zip=zip):
    """Euclidean distance for multidimentional data.

    Args:
        p (Point): First point.
        q (Point): Second point.

    Returns:
        float: Euclidean distance between p and q.
    """
    return sqrt(fsum([(x - y) ** 2 for x, y in zip(p, q)]))


def shift_centroids(centroids, clusters) -> np.ndarray:
    """Shifts centroids to the mean of their clusters.

    Args:
        centroids (Iterable[Centroid]): Iterable of centroids.
        clusters (Iterable[Iterable[Point]]): Iterable of clusters.

    Returns:
        np.ndarray: Array of shifted centroids.
    """
    new_centroids = []
    for centroid in centroids:
        this_cluster = clusters[tuple(centroid)]
        zipped_list = list(zip(*this_cluster))
        # for each tuple in zipped_list, find the mean
        new_centroid = [round(mean(col), 1) for col in zipped_list]
        new_centroids.append(new_centroid)
    return new_centroids


def assign_data(centroids, data) -> dict:
    """Assigns data to centroids and forms clustors.

    Args:
        centroids (Iterable[Centroid]): Iterable of centroids.
        data (Iterable[Point]): Iterable of points.

    Returns:
        np.ndarray: Array of assigned centroids.
    """
    clusters: defaultdict = defaultdict(list)
    for point in data:
        closest_centroid = min(centroids, key=partial(dist, point))
        # convert centroid to tuple to use as key, list or array cannot be used as key
        clusters[tuple(closest_centroid)].append(tuple(point))
    
    return dict(clusters)


def pick_random_centroids(points, k: int):
    """Picks k random points as centroids.

    Args:
        points (Iterable[Point]): Iterable of points.
        k (int): Number of clusters.

    Returns:
        np.ndarray: Array of centroids.
    """
    return random.sample(list(points), k)


def kmeans(points, k: int, iterations: int = 100) -> np.ndarray:
    """K-means clustering.

    Args:
        points (Iterable[Point]): Iterable of points.
        k (int): Number of clusters.
        iterations (int, optional): Number of iterations. Defaults to 100.

    Returns:
        np.ndarray: Array of centroids.
    """
    points = np.array(points)
    centroids = pick_random_centroids(points, k)
    for _ in range(iterations):
        clusters = assign_data(centroids, points)
        centroids = shift_centroids(centroids, clusters)
    return centroids


if __name__ == "__main__":

    iris = datasets.load_iris()

    points: np.ndarray = iris.data

    centroids: np.ndarray = kmeans(points, 3, 50)
 
    d = assign_data(centroids, points)
    pprint(d)