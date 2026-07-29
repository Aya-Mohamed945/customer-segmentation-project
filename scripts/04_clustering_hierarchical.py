
"""
Module: Hierarchical Clustering
Purpose: Apply Hierarchical clustering and visualize dendrogram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_dendrogram(X_scaled, optimal_k, save_path=None):
    """Plot hierarchical clustering dendrogram"""
    # Calculate linkage
    linked = linkage(X_scaled, method='ward')
    
    plt.figure(figsize=(15, 8))
    dendrogram(linked, orientation='top', distance_sort='descending',
               show_leaf_counts=True, truncate_mode='lastp', p=30)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    plt.xlabel('Customer Index')
    plt.ylabel('Distance')
    plt.axhline(y=linked[-optimal_k, 2], color='r', linestyle='--', linewidth=2,
                label=f'Cut at {optimal_k} clusters')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return linked

def apply_hierarchical(X_scaled, n_clusters):
    """Apply Hierarchical clustering"""
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    labels = hierarchical.fit_predict(X_scaled)
    
    print("\n" + "="*60)
    print(f"HIERARCHICAL CLUSTERING RESULTS (k={n_clusters})")
    print("="*60)
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    
    print("\nCluster Distribution:")
    print(pd.Series(labels).value_counts().sort_index())
    
    return hierarchical, labels

if __name__ == "__main__":
    # Load preprocessed data
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    df = pd.read_csv("../data/segmentation_data.csv")
    df_clean = df.drop('ID', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Use optimal k from K-Means
    optimal_k = 9
    
    # Plot dendrogram
    linked = plot_dendrogram(X_scaled, optimal_k, "../results/plots/dendrogram.png")
    
    # Apply Hierarchical
    hierarchical, labels = apply_hierarchical(X_scaled, optimal_k)
    
    print("\nHierarchical clustering completed successfully!")