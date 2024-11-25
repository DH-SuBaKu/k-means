import numpy as np

N = 10 # Number of points
C = 2 # Number of clusters
A,B = -10,1 # Problem Space
np.random.seed(42)

# Create Dataset

dataset = A + (B-A)*np.random.rand(N,2)

dataset = np.array([
    [1, 1], [1.2, 1.1], [0.9, 1.3], [1.1, 0.9], [1.3, 1.2],
    [8, 8], [8.2, 7.9], [7.8, 8.1], [8.1, 8.3], [7.9, 7.8]
])

# Set centroids

centroid_ids = np.random.randint(0,N,size=C)
centroids = dataset[centroid_ids]

# Create clusters

for i in range(1000):

    clusters = [[] for _ in range(C)]

    for i in dataset:
        distances = [np.linalg.norm(i-j) for j in centroids]
        clusters[np.argmin(distances)].append(i)

    centroids = [np.mean(i,axis=0) for i in clusters]

print(centroids)