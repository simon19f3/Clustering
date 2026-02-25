import numpy as np

class DBSCANScratch:
    def __init__(self, eps=0.35, min_samples=15):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        n = len(X)
        labels = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if visited[i]: continue
            visited[i] = True
            
            neighbors = np.where(np.linalg.norm(X - X[i], axis=1) <= self.eps)[0]
            
            if len(neighbors) >= self.min_samples:
                cluster_id = labels.max() + 1 if labels.max() >= 0 else 0
                labels[i] = cluster_id
                
                queue = list(neighbors)
                for idx in queue:
                    if not visited[idx]:
                        visited[idx] = True
                        new_neigh = np.where(np.linalg.norm(X - X[idx], axis=1) <= self.eps)[0]
                        if len(new_neigh) >= self.min_samples:
                            queue.extend([n_idx for n_idx in new_neigh if n_idx not in queue])
                    if labels[idx] == -1:
                        labels[idx] = cluster_id
        self.labels_ = labels
        return self