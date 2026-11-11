import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# first version (v1) had the right idea but the loop structure was broken
# inner while + outer while meant it never properly backtracked
# rewrote everything here as a class to manage state properly and for ease in exporting


class ExtendedKNN:

    def __init__(self, A, w=3.0):
        self.A = A
        self.n = A.shape[0]
        self.w = w
        self.Check    = np.zeros(self.n, dtype=bool)
        self.Labels   = -np.ones(self.n, dtype=int)
        self.Clusters = []
        self.Distance = []


    def threshold(self):
        if len(self.Distance) == 0:
            return np.inf

        # d = np.array(self.Distance)
        # return np.mean(d)

        d = np.array(self.Distance)
        return np.mean(d) + self.w * np.std(d)


    def nearest_unvisited(self, i):

        # min_dist = np.inf
        # min_idx  = -1
        # for j in range(self.n):
        #     if j == i:
        #         continue
        #     if self.Check[j] == True:
        #         continue
        #     temp = np.linalg.norm(self.A[i] - self.A[j])
        #     if temp < min_dist:
        #         min_dist = temp
        #         min_idx  = j
        # return min_idx , min_dist

        dists = np.linalg.norm(self.A - self.A[i], axis=1)
        dists[self.Check] = np.inf
        dists[i] = np.inf
        j = np.argmin(dists)
        return (j, dists[j]) if dists[j] < np.inf else (-1, np.inf)


    def grow(self, start):
        Stack   = [start]
        Cluster = [start]
        self.Check[start] = True

        #  need at least one distance to seed the threshold
        j, d = self.nearest_unvisited(start)
        if j != -1:
            Stack.append(j)
            Cluster.append(j)
            self.Check[j] = True
            self.Distance.append(d)

        # while Stack:
        #     curr = Stack[-1]
        #     j, d = self.nearest_unvisited(curr)
        #     while d < self.threshold():
        #         Stack.append(j)
        #         Cluster.append(j)
        #         self.Check[j] = True
        #         self.Distance.append(d)
        #         curr = j
        #         j, d = self.nearest_unvisited(curr)
        #     Stack.pop()

        while Stack:
            curr = Stack[-1]
            j, d = self.nearest_unvisited(curr)

            if j == -1:
                Stack.pop()
                continue

            if d < self.threshold():
                Stack.append(j)
                Cluster.append(j)
                self.Check[j] = True
                self.Distance.append(d)
            else:
                Stack.pop()   # backtrack

        return Cluster


    def fit(self):
        self.Check    = np.zeros(self.n, dtype=bool)
        self.Labels   = -np.ones(self.n, dtype=int)
        self.Clusters = []
        self.Distance = []
        k = 0

        while not np.all(self.Check):
            unvisited = np.where(~self.Check)[0]

            start = np.random.choice(unvisited)
            C = self.grow(start)
            self.Clusters.append(C)
            self.Labels[C] = k
            k = k + 1

        return self


    def plot(self, title='Extended KNN Clustering'):
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
