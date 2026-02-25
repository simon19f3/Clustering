import pandas as pd
import matplotlib.pyplot as plt
from src.kmeans import KMeansScratch
from src.dbscan import DBSCANScratch
from src.hdbscan import HDBSCANScratch
from src.metrics import get_all_metrics

# Load Data
df = pd.read_csv("clustering_data.csv")
X = (df.values - df.values.mean(axis=0)) / df.values.std(axis=0)

# Run Algorithms
models = {
    "K-Means": KMeansScratch(k=3).fit(X),
    "DBSCAN": DBSCANScratch(eps=None, min_samples=20).fit(X),
    "HDBSCAN": HDBSCANScratch(min_cluster_size=20).fit(X)
}

# Evaluation
results = [get_all_metrics(X, model.labels_, name) for name, model in models.items()]
print(pd.DataFrame(results))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, model) in enumerate(models.items()):
    axes[i].scatter(X[:, 0], X[:, 1], c=model.labels_, s=2, cmap='viridis')
    axes[i].set_title(name)
plt.tight_layout()
plt.savefig("clustering_results.png")
plt.show()