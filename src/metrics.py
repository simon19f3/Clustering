import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def silhouette_from_scratch(X, labels, sample_size=1000):
    mask = labels != -1
    X_f, L_f = X[mask], labels[mask]
    unique_labels = np.unique(L_f)
    if len(unique_labels) < 2: return 0
    
    indices = np.random.choice(len(X_f), min(sample_size, len(X_f)), replace=False)
    scores = []
    for i in indices:
        a = np.mean(np.linalg.norm(X_f[L_f == L_f[i]] - X_f[i], axis=1))
        b = min([np.mean(np.linalg.norm(X_f[L_f == l] - X_f[i], axis=1)) 
                 for l in unique_labels if l != L_f[i]])
        scores.append((b - a) / max(a, b))
    return np.mean(scores)

def get_all_metrics(X, labels, name):
    mask = labels != -1
    if len(np.unique(labels[mask])) < 2:
        return {"Algorithm": name, "Status": "Insufficient clusters"}
    
    return {
        "Algorithm": name,
        "Silh (Scratch)": silhouette_from_scratch(X, labels),
        "Silh (Built-in)": silhouette_score(X[mask], labels[mask]),
        "Davies-Bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "Calinski-Harabasz": calinski_harabasz_score(X[mask], labels[mask])
    }