# Clustering Analysis from Scratch 🚀

A comprehensive implementation of core clustering algorithms—**K-Means**, **DBSCAN**, and **HDBSCAN**—built entirely from scratch using only NumPy and Pandas.  
This project demonstrates the mathematical logic behind these algorithms and evaluates their performance on a high-volume dataset (10,000+ samples).

## 🛠 Features

- **Algorithms from Scratch**:
  - K-Means: Implementation with K-means++ initialization
  - DBSCAN: Density-based clustering with an automated epsilon (ε) estimation heuristic
  - HDBSCAN: Hierarchical density clustering using Mutual Reachability Distance and Prim's Minimum Spanning Tree (MST)

- **Evaluation Suite**:
  - Silhouette Score (custom implementation vs. scikit-learn)
  - Davies-Bouldin Index
  - Calinski-Harabasz Index

- **Vectorized Performance**: Optimized for 10k+ data points using NumPy broadcasting

## 📂 Project Structure

```text
├── src/
│   ├── kmeans.py       # Centroid-based clustering logic
│   ├── dbscan.py       # Density-based logic with auto-eps heuristic
│   ├── hdbscan.py      # Hierarchical density clustering (MST-based)
│   └── metrics.py      # Custom and built-in evaluation functions
├── main.py             # Entry point for data loading, training, and plotting
├── clustering_dataset.csv # Dataset (approx. 10,000 samples)
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
