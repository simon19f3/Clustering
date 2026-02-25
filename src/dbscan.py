import numpy as np

class DBSCANScratch:
    def __init__(self, eps=None, min_samples=10):
        self.eps = eps
        self.min_samples = min_samples

    def _estimate_eps(self, dist_matrix):
        """
        Pure NumPy version of the k-distance heuristic.
        Finds the distance to the k-th neighbor for every point.
        """
        # Sort distances for each point (axis=1) and pick the distance 
        # to the k-th neighbor (k = min_samples)
        # np.partition is faster than np.sort for finding specific ranks
        k_distances = np.partition(dist_matrix, self.min_samples, axis=1)[:, self.min_samples]
        
        # Sort the k-distances and pick the 90th percentile as the "elbow"
        k_distances_sorted = np.sort(k_distances)
        return np.percentile(k_distances_sorted, 90)

    def fit(self, X):
        n = len(X)
        
        # 1. Precompute distance matrix ONCE (approx 800MB for 10k points)
        # Using vectorized subtraction and squaring is faster than np.linalg.norm in some NumPy versions
        dist_matrix = np.sqrt(np.sum((X[:, np.newaxis] - X)**2, axis=2))

        # 2. Auto-select epsilon if not provided
        if self.eps is None:
            self.eps = self._estimate_eps(dist_matrix)
            print(f"DBSCAN: Auto-selected eps = {self.eps:.4f}")

        labels = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        # 3. Clustering Logic
        for i in range(n):
            if visited[i]:
                continue

            visited[i] = True
            # Use the precomputed matrix instead of recalculating
            neighbors = np.where(dist_matrix[i] <= self.eps)[0]

            if len(neighbors) < self.min_samples:
                continue

            # Start new cluster
            labels[i] = cluster_id
            seeds = list(neighbors)

            # Cluster expansion
            while seeds:
                current = seeds.pop()
                if not visited[current]:
                    visited[current] = True
                    current_neighbors = np.where(dist_matrix[current] <= self.eps)[0]

                    if len(current_neighbors) >= self.min_samples:
                        # Extend the seeds list (only add new neighbors not already visited)
                        for nb in current_neighbors:
                            if not visited[nb]:
                                seeds.append(nb)

                if labels[current] == -1:
                    labels[current] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        return self