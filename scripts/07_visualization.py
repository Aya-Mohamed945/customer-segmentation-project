"""
Module: Visualization
Purpose: Visualize clustering results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_cluster_comparison(X_scaled, kmeans_labels, hier_labels, dbscan_labels=None, 
                            optimal_k=9, best_params=None, save_path=None):
    """Plot comparison of clustering algorithms"""
    # Apply PCA for 2D visualization
    pca_vis = PCA(n_components=2)
    X_pca_vis = pca_vis.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Comparison of Clustering Algorithms', fontsize=16, fontweight='bold')
    
    # K-Means
    scatter1 = axes[0].scatter(X_pca_vis[:, 0], X_pca_vis[:, 1],
                              c=kmeans_labels, cmap='viridis', alpha=0.7, s=40)
    axes[0].set_title(f'K-Means (k={optimal_k})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0])
    
    # Hierarchical
    scatter2 = axes[1].scatter(X_pca_vis[:, 0], X_pca_vis[:, 1],
                              c=hier_labels, cmap='plasma', alpha=0.7, s=40)
    axes[1].set_title('Hierarchical Clustering', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1])
    
    # DBSCAN
    if dbscan_labels is not None and best_params is not None:
        scatter3 = axes[2].scatter(X_pca_vis[:, 0], X_pca_vis[:, 1],
                                  c=dbscan_labels, cmap='rainbow', alpha=0.7, s=40)
        axes[2].set_title(f'DBSCAN (eps={best_params[0]:.1f}, min_samples={best_params[1]})', 
                         fontsize=12, fontweight='bold')
        axes[2].set_xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}%)')
        axes[2].set_ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}%)')
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return X_pca_vis, pca_vis

def plot_kmeans_with_centroids(X_scaled, kmeans_labels, kmeans_centroids, save_path=None):
    """Plot K-Means clusters with centroids"""
    # Apply PCA for 2D visualization
    pca_vis = PCA(n_components=2)
    X_pca_vis = pca_vis.fit_transform(X_scaled)
    centroids_pca = pca_vis.transform(kmeans_centroids)
    
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    scatter = plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1],
                         c=kmeans_labels, cmap='viridis', alpha=0.6, s=30)
    
    # Plot centroids
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c='red', marker='X', s=300, edgecolors='black', linewidths=2,
               label='Cluster Centers', zorder=5)
    
    n_clusters = len(np.unique(kmeans_labels))
    plt.title(f'K-Means Clustering Results (k={n_clusters})', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}%)')
    plt.colorbar(scatter, label='Cluster Number')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return X_pca_vis, pca_vis

def plot_cluster_characteristics(df_clustered, save_path=None):
    """Plot cluster characteristics using boxplots"""
    features_to_plot = ['Age', 'Income', 'Sex', 'Marital status', 'Education', 'Occupation']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Cluster Characteristics Analysis', fontsize=16, fontweight='bold')
    
    for idx, feature in enumerate(features_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        df_clustered.boxplot(column=feature, by='Cluster_KMeans', ax=ax)
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cluster Number')
        ax.set_ylabel(f'{feature}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # This is a utility module - will be called from main notebook
    print("Visualization module loaded successfully!")
