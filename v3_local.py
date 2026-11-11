import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles


# v2 kept Distance as a global list across all clusters
# tested it on moons and it worked but i noticed something odd:
# when i ran it on a dataset with two regions of very different density
# the dense cluster would dominate the Distance list early on
# and then the threshold for the sparse cluster was set way too tight
# it would fragment the sparse cluster into dozens of tiny pieces
#
# fix: reset Distance at the start of each new cluster
# that way each cluster calibrates its own threshold from scratch
# trade-off is that the very first step of each cluster is always unconditional
# (since Distance is empty) — but that was already the case in v2


#
# def grow(self, start):
#     Stack   = [start]
#     Cluster = [start]
#     self.Check[start] = True
#
#     j, d = self.nearest_unvisited(start)
#     if j != -1:
#         Stack.append(j); Cluster.append(j)
#         self.Check[j] = True
#         self.Distance.append(d)        # <-- adds to the shared global list
#
#     while Stack:
#         curr = Stack[-1]
#         j, d = self.nearest_unvisited(curr)
#         if j == -1: Stack.pop(); continue
#         if d < self.threshold():
#             Stack.append(j); Cluster.append(j)
#             self.Check[j] = True
#             self.Distance.append(d)    # <-- same global list
#         else:
#             Stack.pop()
#     return Cluster
#
# and in fit(), Distance was initialised once and never reset
# ------------------------------------------------------------


class ExtendedKNN_Local:

    def __init__(self, A, w=3.0):
        self.A = A
        self.n = A.shape[0]
        self.w = w
        self.Check   = np.zeros(self.n, dtype=bool)
        self.Labels  = -np.ones(self.n, dtype=int)
        self.Clusters = []

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
        Distance = []    # local to this cluster — reset every time

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

    def plot(self, title='Extended KNN — Local Density'):
        plt.figure(figsize=(9, 6))
        for label in np.unique(self.Labels):
            mask = self.Labels == label
            plt.scatter(self.A[mask, 0], self.A[mask, 1], s=40, alpha=0.7,
                        edgecolors='k', linewidth=0.4, label=f'C{label}')
        plt.title(title)
        plt.legend(fontsize=7)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
