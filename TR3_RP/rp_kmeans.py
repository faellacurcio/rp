import numpy
import random
import math
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt


SAMPLES = list()
# Abre e lê iris_log como um DATASET
with open('iris_log.dat') as iris:
    for row in iris.readlines():
        SAMPLES.append(list(map(float, row.split())))
DATASET = numpy.array(SAMPLES)
# separa os dados em features e classe (X/Y)

X = DATASET[:, :-3]


def dist(vec_a, vec_b):
    # Distancia Euclidiana
    return numpy.linalg.norm(numpy.array([vec_a]) - vec_b, axis=1)


silhuheta = []
for num_centroids in range(2, 8):

    centroids = random.choices(X, k=num_centroids)

    print("Initial Centroids")
    print(centroids)

    old_centroids = numpy.zeros(X.shape[1])
    clusters = numpy.zeros(len(X))
    error = dist(centroids, old_centroids)

    while True:
        for idx in range(len(X)):
            distances = dist(X[idx], centroids)
            cluster = numpy.argmin(distances)
            clusters[idx] = cluster
        # Storing the old centroid values
        old_centroids = deepcopy(centroids)
        # Finding the new centroids by taking the average value
        for idx in range(num_centroids):
            points = [X[j] for j in range(len(X)) if clusters[j] == idx]
            centroids[idx] = numpy.mean(points, axis=0)
        error = dist(centroids, old_centroids)
        if(sum(sum(error)) < 0.1):
            break

    closest_dist = math.inf*numpy.ones(num_centroids)
    closest_idx = numpy.array([list(range(num_centroids))]).T

    for idx, _ in enumerate(centroids):
        for idx2, centroid in enumerate(centroids):
            if(idx != idx2 and closest_dist[idx] > dist(centroids[idx], centroids[idx2])):
                closest_dist[idx] = dist(centroids[idx], centroids[idx2])
                closest_idx[idx] = idx2

    a_vec = numpy.zeros([1, len(X[:, 0])]).T
    b_vec = numpy.zeros([1, len(X[:, 0])]).T

    for i in range(len(X[:, 0])):
        a_ith_mean = numpy.array([0.0])
        b_ith_mean = numpy.array([0.0])
        counter_a = 0
        counter_b = 0
        for idx in range(len(X[:, 0])):
            if(i != idx and clusters[i] == clusters[idx]):
                a_ith_mean += dist(X[i], X[idx])
                counter_a += 1
            elif(clusters[i] == closest_idx[int(clusters[idx])]):
                b_ith_mean += dist(X[i], X[idx])
                counter_b += 1
        if(counter_a != 0):
            a_vec[i] = a_ith_mean/counter_a
        if(counter_b != 0):
            b_vec[i] = b_ith_mean/counter_b

    print("Silhueta:")
    print((numpy.mean(b_vec)-numpy.mean(a_vec))/(max(max(a_vec), max(b_vec))))
    silhuheta.append((numpy.mean(b_vec)-numpy.mean(a_vec)) /
                     (max(max(a_vec), max(b_vec))))
print([float(x) for x in silhuheta])
