# -*- coding: utf-8 -*-


"""
Module: K-Means Clustering
Purpose: Apply K-Means and determine optimal k
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Find optimal number of clusters using multiple metrics"""
    inertia = []
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
        davies_scores.append(davies_bouldin_score(X_scaled, labels))
    
    # Find optimal
    optimal_k_sil = k_range[np.argmax(silhouette_scores)]
    optimal_k_cal = k_range[np.argmax(calinski_scores)]
    optimal_k_db = k_range[np.argmin(davies_scores)]
    
    # Print results
    print("\n" + "="*60)
    print("CLUSTER EVALUATION RESULTS")
    print("="*60)
    results_df = pd.DataFrame({
        'k': k_range,
        'Inertia': inertia,
        'Silhouette': silhouette_scores,
        'Calinski-Harabasz': calinski_scores,
        'Davies-Bouldin': davies_scores
    })
    print(results_df.round(4))
    
    print(f"\nOptimal k based on:")
    print(f"  • Silhouette Score: {optimal_k_sil}")
    print(f"  • Calinski-Harabasz: {optimal_k_cal}")
    print(f"  • Davies-Bouldin: {optimal_k_db}")
    
    return optimal_k_sil, inertia, silhouette_scores, calinski_scores, davies_scores

def plot_evaluation_metrics(k_range, inertia, silhouette_scores, calinski_scores, davies_scores, optimal_k, save_path=None):
    """Plot clustering evaluation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimal Number of Clusters - Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # 1. Elbow Method
    axes[0, 0].plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia (Within-Cluster SSE)')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Silhouette Score
    axes[0, 1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].axvline(x=optimal_k, color='g', linestyle='--', alpha=0.5, label=f'Optimal k={optimal_k}')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Score (Higher is Better)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Calinski-Harabasz Score
    axes[1, 0].plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title('Calinski-Harabasz Index (Higher is Better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Davies-Bouldin Score
    axes[1, 1].plot(k_range, davies_scores, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Davies-Bouldin Score')
    axes[1, 1].set_title('Davies-Bouldin Index (Lower is Better)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def apply_kmeans(X_scaled, n_clusters):
    """Apply K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    
    print("\n" + "="*60)
    print(f"K-MEANS CLUSTERING RESULTS (k={n_clusters})")
    print("="*60)
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
    
    print("\nCluster Distribution:")
    print(pd.Series(labels).value_counts().sort_index())
    
    return kmeans, labels, centroids

if __name__ == "__main__":
    # Load preprocessed data
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    df = pd.read_csv("../data/segmentation_data.csv")
    df_clean = df.drop('ID', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Find optimal k
    optimal_k, inertia, silhouette, calinski, davies = find_optimal_k(X_scaled)
    
    # Plot evaluation metrics
    plot_evaluation_metrics(
        range(2, 11), inertia, silhouette, calinski, davies, 
        optimal_k, "../results/plots/evaluation_metrics.png"
    )
    
    # Apply K-Means
    kmeans, labels, centroids = apply_kmeans(X_scaled, optimal_k)
    
    print("\nK-Means clustering completed successfully!")
