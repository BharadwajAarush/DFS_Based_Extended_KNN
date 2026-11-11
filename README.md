# DFS-Based Extended KNN Clustering

**Author:** Aarush Bharadwaj  
**Status:** Research in Progress 

## Motivation

Standard K-Means clustering assumes that clusters are **convex and globular**  i.e sort of spherical blobs of similar size. This assumption breaks badly on real-world data where clusters form **non-convex, manifoldesque shapes**: crescents, rings, chains, or irregular regions eg the type of geometry usually encountered in medical image segmentation and materials science

This project explores a DFS-inspired clustering algorithm designed to follow a type of local neighbourhood connectivity without the user needing to put in a predefined number of clusters(i.e K).

---

## The Algorithm

### Core Idea

While working on and implementing some fundamental clustering algorithms , I had a philosphical question , what makes a point a part of a cluster ? we can imagine some 10 parallel rods kept at a distance of 10m from each other and let each rod be 12m long , then if asked to cluster the points in this setup we will put all the points on a rod in a single cluster , despite the distance between the point on the top of the rod and the bottom of the rod is more than distance between the point on the bottom of the rod with the point on the bottom of the next adjacent rod, 
So it made me clear , it is not the Distance but it is the proximity of connected NEIGHBOURS which makes a point belong to a certain cluster , 
building upon this I tried having an algorithm which would start from a point and find all it's nearest (unvisited )neighbours and keep them in a stack and then go the next neighbour with minimum distance and yet again find all the nearest neighbour points ( apart from the one , which we have already traversed ) and then go to the next neighbour , Evidently it feels like a traversing algo which would traverse all the points in the setup and how would it be used for clustering ? , but our traversal shall have some contraints and they are- 
    1- We shall never hop on to a neighbour if the distance between the current point and the said neighbour is more than D ( here this D is calculated with the help of mean of all the distance and added with a W(weight) * multiplied by the S.D of the distances , mean(distances) + w·std(distances ; hence it is evident if W is large then the said clusters would be big and not many probable points would be rejected from being the Neighbour))
    2- If a said point has no new neighbours , we shall start exhausting all the points in the said stack 
    3- If the stack is empty , we shall call the said traversal of all the points a cluster and consequently remove them from the net available points 
Now for the algo to start we shall sleect a point at randon and evidently , if we get an outlier point in the first iteration then all the points which are it's neighbour , would be at a high distance and hence the distance matrix which are maintaining wo calculate D , would sorta take a plummit and the said point would now be part of some cluster , it is one of the limitations of the algo , I have thougt of one way to imporve on this , by empircally calculating the AVG distances between some clusters by running some other algos on it first 

This algo comes with many advantages and the best one being is , you need not tell it to cluster the data into how many clusters , it would go through the data and it would tell itself how many clusters are optimizal in the said case 



I have added a semi informal youtube video in Hinglish(with english translation) to explain it -->https://www.youtube.com/watch?v=ZE7aCYOHqxk

### Algorithm (Pseudocode)

```
Input: Data matrix A (n × d), weight parameter w
Output: Cluster labels for all n points

Distance ← []          # Global accepted distance history
Check    ← [False] * n # Visited indicator
Labels   ← [-1] * n

while any point unvisited:
    p ← random unvisited point
    Stack ← [p],  Cluster ← [p],  Check[p] ← True

    # Unconditional first step: seeds the distance memory to compute threshold
    q, d ← nearest_unvisited(p)
    if q exists:
        Stack.append(q), Cluster.append(q)
        Check[q] ← True
        Distance.append(d)

    # DFS traversal loop
    while Stack not empty:
        current ← Stack.top()
        q, d    ← nearest_unvisited(current)

        if q does not exist:
            Stack.pop()                         # Backtrack when the nearest unvisited candidate is rejected
        elif d < mean(Distance) + w · std(Distance):
            Stack.append(q), Cluster.append(q)  # Accept hop and go deeper
            Check[q] ← True
            Distance.append(d)
        else:
            Stack.pop()                         # Backtrack to find alternate branch

    assign new cluster ID to all points in Cluster
```

### The Key Innovation: Self-Calibrating Threshold

Unlike DBSCAN which requires a fixed epsilon $\varepsilon$, this algorithm computes its distance threshold **online** from the history of distances it has already accepted(i.e on the GO):

$$\text{threshold} = \text{mean}(\text{Distance}) + w \cdot \text{std}(\text{Distance})$$

The threshold is updated using the global history of accepted neighbour distances accumulated throughout the traversal. The parameter $w$ controls how permissive the acceptance criterion is.

---

## Repository Structure

```
DFS_Based_Extended_KNN/
│
├── README.md                  ← Project documentation, pseudocode, and benchmark results
│
├── v1_original.py             ← Initial prototype (nested while-loop formulation)
├── v2_global.py               ← Canonical Extended KNN (global density learning)
├── v3_local.py                ← Local density variant (resets distance memory per cluster)
├── v4_spectral.py             ← Spectral projection pipeline for higher dimensions
│
└── notebook/
    ├── demo.ipynb              ← K-Means vs Extended KNN comparison (Moons, Circles, Blobs)
    └── Extended_KNN_Demo.ipynb ← Step-by-step tutorial, diagnostic plots, and weight tuning
```

---

## Algorithm Variants

Three variants address different data distributions:

### 1. Global Density (`v2_global.py`) — *Canonical Implementation*
A **single distance history** accumulates across all clusters:
$$\text{threshold} = \mu(\mathcal{D}) + w \cdot \sigma(\mathcal{D})$$
The threshold reflects dataset-wide connectivity. Best for datasets with relatively uniform density (moons, rings, chains).

### 2. Local Density (`v3_local.py`)
Distance history **resets for each new cluster** ($\mathcal{D}_k = []$). Each cluster now gets to learns its own local density threshold independently from scratch. Best suited when different clusters have significantly different densities or internal compactness.

### 3. Spectral Projection (`v4_spectral.py`)
For high-dimensional feature spaces: first projects data onto a low-dimensional manifold using **Spectral Embedding (Laplacian Eigenmaps)**, The work is also under progress in this Algo.
---

## Development History

The algorithm was developed iteratively under research guidance from Dr. Dhirendra Kumar:

| Version | File | Key Formulation & Focus |
|---|---|---|
| v1 | `v1_original.py` | Initial prototype exploring DFS stack and online $(\mu + w \cdot \sigma)$ threshold. |
| v2 | `v2_global.py` | Canonical class-based implementation with global distance memory and clean backtracking. |
| v3 | `v3_local.py` | Per-cluster density reset to accommodate multi-density datasets. |
| v4 | `v4_spectral.py` | Spectral embedding pipeline to preserve non-linear geometry in higher dimensions. |

---

## Benchmark Results

### Synthetic Shape Comparison (Demo Suite, w = 3.5)

| Dataset | Geometry | True K | K-Means ARI | V2 Found K | V2 ARI |
|---|---|---|---|---|---|
| Moons | Non-convex | 2 | 0.479 | 3 | **0.987** |
| Circles | Non-convex | 2 | -0.003 | 3 | **0.987** |
| Blobs | Convex | 3 | **1.000** | 5 | 0.980 |

**Observations**:
V2 recovers the principal non-convex structure in Moons and Circles where K-Means fails due to its spherical/linear partition assumption. On convex Blobs, K-Means achieves a higher ARI and exact cluster count ($K=3$), while V2 produces minor small auxiliary clusters due to the unconditional initial neighbour seeding.

> **To reproduce:** Run `notebook/demo.ipynb` or `notebook/Extended_KNN_Demo.ipynb` — all benchmark data is generated self-contained without external downloads.

---

## Ongoing Work

- **Mathematical convergence analysis** — proving asymptotic stability of the running mean and variance under manifold sampling assumptions.
- **Outlier rejection** — refining the unconditional initial neighbour selection to prevent outlier seed contamination.
- **Connection to MRI segmentation** — applying topological manifold tracking to irregular lesion and tissue boundary detection.

---

## Quick Start

```bash
pip install numpy matplotlib scikit-learn
```

Run directly in Python:
```python
import numpy as np
from v2_global import ExtendedKNN

# Load or generate your data (n_samples, n_features)
X = ...

# Initialize and fit
model = ExtendedKNN(X, w=3.5)
model.fit()

# Discovered clusters and assignments
print("Clusters found:", len(model.Clusters))
print("Point labels:", model.Labels)
```

Interactive notebooks:
- `notebook/demo.ipynb`: Side-by-side comparison with K-Means across non-convex and convex benchmarks.
- `notebook/Extended_KNN_Demo.ipynb`: Step-by-step walkthrough, global density learning diagnostic plots, and weight tuning.

---



## Note
Some of the formatting in this said ReadME and Project has been done using OLlama (So , a special thanks to Ollama)
