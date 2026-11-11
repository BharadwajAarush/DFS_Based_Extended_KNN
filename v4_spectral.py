import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# this direction is still under progress 
# both v2 and v3 work well on 2D data with clear geometric structure
# tested on Iris (4D) and Digits (64D) — results were poor, ARI close to 0
#
# the problem: in high dimensions, the nearest-unvisited-neighbor step becomes meaningless because
# euclidean distance loses its geometric sense (cuz of curse of dimensionality )
# so the DFS walk just wanders randomly and the threshold never gets to stabilise properly
#
# idea: project the data to a low-dimensional manifold first
# used Laplacian Eigenmaps (SpectralEmbedding in sklearn)
# which builds a nearest-neighbour graph and embeds it into k dimensions
# preserving local neighbourhood structure
# then run the same algorithm on that embedding
#
# one thing i added that wasn't in v2/v3:
# normalise the embedding with StandardScaler before running
# without this the threshold scale depends on the embedding's arbitrary units
# normalising makes w=3.0 behave consistently regardless of the dataset


class ExtendedKNN_Spectral:

    def __init__(self, A, w=3.0, n_neighbors=10, n_components=2):
        self.A_raw       = A
        self.w           = w
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.n           = A.shape[0]
        self.A           = None   # will hold the embedding
        self.Check       = np.zeros(self.n, dtype=bool)
        self.Labels      = -np.ones(self.n, dtype=int)
        self.Clusters    = []

    def embed(self):
        emb  = SpectralEmbedding(n_components=self.n_components,
                                  n_neighbors=self.n_neighbors,
                                  affinity='nearest_neighbors')
        proj = emb.fit_transform(self.A_raw)
        self.A = StandardScaler().fit_transform(proj)   # normalise

    def threshold(self, Distance):
        if len(Distance) == 0:
            return np.inf
        d = np.array(Distance)
        return np.mean(d) + self.w * np.std(d)

    def nearest_unvisited(self, i):
        dists = np.linalg.norm(self.A - self.A[i], axis=1)
        dists[self.Check] = np.inf
        dists[i] = np.inf
        j = np.argmin(dists)
        return (j, dists[j]) if dists[j] < np.inf else (-1, np.inf)

    def grow(self, start):
        Distance = []

        Stack   = [start]
        Cluster = [start]
        self.Check[start] = True

        j, d = self.nearest_unvisited(start)
        if j != -1:
            Stack.append(j)
            Cluster.append(j)
            self.Check[j] = True
            Distance.append(d)

        while Stack:
            curr = Stack[-1]
            j, d = self.nearest_unvisited(curr)

            if j == -1:
                Stack.pop()
                continue

            if d < self.threshold(Distance):
                Stack.append(j)
                Cluster.append(j)
                self.Check[j] = True
                Distance.append(d)
            else:
                Stack.pop()

        return Cluster

    def fit(self):
        self.embed()
        self.Check    = np.zeros(self.n, dtype=bool)
        self.Labels   = -np.ones(self.n, dtype=int)
        self.Clusters = []
        k = 0

        while not np.all(self.Check):
            unvisited = np.where(~self.Check)[0]
            start = np.random.choice(unvisited)
            C = self.grow(start)
            self.Clusters.append(C)
            self.Labels[C] = k
            k += 1

        return self

