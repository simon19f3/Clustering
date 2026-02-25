import numpy as np

class KMeansScratch:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        # K-means++ initialization
        idx = [np.random.randint(len(X))]
        for _ in range(1, self.k):
            dists = np.min([np.sum((X - X[i])**2, axis=1) for i in idx], axis=0)
            idx.append(np.argmax(dists))
        
        self.centroids = X[idx]
        
        for _ in range(self.max_iter):
            # Assignment
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) 
                                      else self.centroids[i] for i in range(self.k)])
            
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
            
        self.labels_ = labels
        return self