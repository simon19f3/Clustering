import numpy as np

class HDBSCANScratch:
    def __init__(self, min_cluster_size=15):
        self.min_cluster_size = min_cluster_size

    def fit(self, X):
        n = len(X)
        # 1. Mutual Reachability Distance
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        core_dists = np.partition(dist_matrix, self.min_cluster_size, axis=1)[:, self.min_cluster_size]
        mrd = np.maximum(dist_matrix, np.maximum(core_dists[:, np.newaxis], core_dists))
        
        # 2. Prim's MST (O(n^2))
        min_dists, parent, visited = np.full(n, np.inf), np.full(n, -1), np.zeros(n, dtype=bool)
        min_dists[0], mst = 0, []
        for _ in range(n):
            u = np.argmin(np.where(visited, np.inf, min_dists))
            visited[u] = True
            if parent[u] != -1: mst.append([parent[u], u, min_dists[u]])
            
            mask = ~visited
            new_dists = mrd[u, mask]
            update = new_dists < min_dists[mask]
            indices = np.where(mask)[0][update]
            min_dists[indices], parent[indices] = new_dists[update], u
        
        # 3. Simple Extraction via Union-Find
        mst = np.array(mst)
        mst = mst[mst[:, 2].argsort()]
        threshold = np.percentile(mst[:, 2], 85)
        
        parent_arr = np.arange(n)
        def find(i):
            while parent_arr[i] != i: i = parent_arr[i]
            return i

        for u, v, w in mst:
            if w <= threshold:
                root_u, root_v = find(int(u)), find(int(v))
                if root_u != root_v: parent_arr[root_u] = root_v
        
        labels = np.full(n, -1)
        curr_id = 0
        unique_roots = np.unique([find(i) for i in range(n)])
        for r in unique_roots:
            members = [i for i in range(n) if find(i) == r]
            if len(members) >= self.min_cluster_size:
                labels[members] = curr_id
                curr_id += 1
        self.labels_ = labels
        return self