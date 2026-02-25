# Clustering From Scratch

Implementation of K-Means, DBSCAN, and HDBSCAN using only NumPy and Pandas.

## Analysis
- **K-Means**: Best for spherical clusters. Highest Calinski-Harabasz score.
- **DBSCAN**: Effectively removes noise (labeled as -1). Requires precise `eps` tuning.
- **HDBSCAN**: Best for clusters with varying densities. Most complex implementation.

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Run analysis: `python main.py`