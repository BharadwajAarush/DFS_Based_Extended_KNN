# This is the first attempt at implementing the algorithm.
# It is an attempt to understand the core structure of the Algo and how updates shall happen and where 
# The core logic (DFS + stack + mean/std threshold) is all here, just not clean.

import numpy as np
import matplotlib.pyplot as plt 


class Stack:
    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)

    def pop(self):
        
        if self.isempty():
            raise IndexError("Pop from empty stack")
        return self._data.pop()

    def isempty(self):
        return len(self._data) == 0


Stack = Stack()
Distance = []

n = A.shape[0]
Check = np.ones(n, dtype=bool)

Clusters = []


def mean_and_std(values):
    w = 1
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    std = np.std(values)
    return mean + w*std

def sorted_distances(A, i):
    diff = A - A[i]
    distances = np.linalg.norm(diff, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    return sorted_distances

def sorted_indices(A, i):
    diff = A - A[i]
    distances = np.linalg.norm(diff, axis=1)
    sorted_indices = np.argsort(distances)
    return sorted_indices


A = np.random.rand(n, d)

p = np.random.randint(n)
Clusters.append(p)
Stack.append(p)
Check[p] = 0

curr = p
while(!Stack.empty()):
    while(sorted_distances(p)[0] < mean_and_std(Distance) and Check[p]):
        Distances.append(sorted_distances(curr)[0])
        curr = sorted_indices(A, curr)
        Clusters.append(sorted_indices(A, curr))
        Stack.push(curr)
        Check[curr] = [0]

    curr = stack.pop()
