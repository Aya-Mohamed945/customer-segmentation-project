"""
Module: Model Evaluation
Purpose: Compare clustering algorithms and evaluate results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def evaluate_algorithms(X_scaled, kmeans_labels, hier_labels, dbscan_labels=None):
    """Evaluate and compare clustering algorithms"""
    evaluation_data = {
        'Algorithm': ['K-Means', 'Hierarchical', 'DBSCAN'],
        'Silhouette Score': [
            silhouette_score(X_scaled, kmeans_labels),
            silhouette_score(X_scaled, hier_labels),
            silhouette_score(X_scaled, dbscan_labels) if dbscan_labels is not None else np.nan
        ],
        'Number of Clusters': [
            len(np.unique(kmeans_labels)),
            len(np.unique(hier_labels)),
            len(np.unique(dbscan_labels)) if dbscan_labels is not None else np.nan
        ]
    }
    
    evaluation_df = pd.DataFrame(evaluation_data)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*60)
    print(evaluation_df.round(4))
    
    return evaluation_df

def print_algorithm_comparison():
    """Print advantages and disadvantages of each algorithm"""
    print("\n" + "="*60)
    print("ADVANTAGES & DISADVANTAGES OF EACH ALGORITHM")
    print("="*60)
    
    print("\n✅ K-MEANS CLUSTERING:")
    print("   Advantages:")
    print("   • Fast and efficient for large datasets")
    print("   • Easy to interpret and implement")
    print("   • Guarantees convergence")
    print("   • Works well with spherical clusters")
    print("   Disadvantages:")
    print("   • Requires specifying number of clusters (k)")
    print("   • Sensitive to initial centroids")
    print("   • Assumes spherical cluster shape")
    print("   • Affected by outliers")
    
    print("\n✅ HIERARCHICAL CLUSTERING:")
    print("   Advantages:")
    print("   • No need to specify number of clusters in advance")
    print("   • Provides dendrogram for visualization")
    print("   • Produces hierarchical relationships")
    print("   • Deterministic (same results each run)")
    print("   Disadvantages:")
    print("   • Computationally expensive for large datasets (O(n³))")
    print("   • Sensitive to noise and outliers")
    print("   • Once decisions made, cannot be undone")
    
    print("\n✅ DBSCAN CLUSTERING:")
    print("   Advantages:")
    print("   • Does not require specifying number of clusters")
    print("   • Can find arbitrarily shaped clusters")
    print("   • Robust to outliers (identifies noise points)")
    print("   • Works well with non-linear clusters")
    print("   Disadvantages:")
    print("   • Requires tuning parameters (eps and min_samples)")
    print("   • Struggles with varying density clusters")
    print("   • Sensitive to parameter selection")

if __name__ == "__main__":
    # This is a utility module - will be called from main notebook
    print("Model evaluation module loaded successfully!")