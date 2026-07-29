"""
Module: DBSCAN Clustering
Purpose: Apply DBSCAN and find optimal parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

def plot_k_distance_graph(X_scaled, save_path=None):
    """Plot k-distance graph for DBSCAN"""
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    distances_sorted = np.sort(distances[:, 4])
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances_sorted, 'b-', linewidth=2)
    plt.xlabel('Data Points (sorted)')
    plt.ylabel('Distance to 5th Nearest Neighbor')
    plt.title('k-Distance Graph for DBSCAN', fontsize=14, fontweight='bold')
    plt.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='eps ≈ 2.0')
    plt.axhline(y=2.5, color='g', linestyle='--', alpha=0.5, label='eps ≈ 2.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def tune_dbscan(X_scaled, eps_values, min_samples_values):
    """Tune DBSCAN parameters"""
    best_score = -1
    best_params = None
    best_labels = None
    
    print("\n" + "="*60)
    print("DBSCAN PARAMETER TUNING")
    print("="*60)
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters > 1 and n_clusters <= 8:
                try:
                    score = silhouette_score(X_scaled, labels)
                    print(f"eps={eps:.1f}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise points, Silhouette={score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)
                        best_labels = labels
                except:
                    print(f"eps={eps:.1f}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise points - Cannot compute Silhouette")
    
    if best_params is not None:
        print(f"\nBest DBSCAN parameters: eps={best_params[0]:.1f}, min_samples={best_params[1]}")
        print(f"Silhouette Score: {best_score:.4f}")
    
    return best_params, best_labels, best_score

def apply_dbscan(X_scaled, eps, min_samples):
    """Apply DBSCAN with given parameters"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print("\n" + "="*60)
    print(f"DBSCAN CLUSTERING RESULTS (eps={eps:.1f}, min_samples={min_samples})")
    print("="*60)
    print(f"Number of clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    
    print("\nCluster Distribution:")
    print(pd.Series(labels).value_counts().sort_index())
    
    return dbscan, labels

if __name__ == "__main__":
    # Load preprocessed data
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    df = pd.read_csv("../data/segmentation_data.csv")
    df_clean = df.drop('ID', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Plot k-distance graph
    plot_k_distance_graph(X_scaled, "../results/plots/dbscan_k_distance.png")
    
    # Tune parameters
    eps_values = [1.5, 1.8, 2.0, 2.2, 2.5]
    min_samples_values = [3, 5, 7, 10]
    best_params, best_labels, best_score = tune_dbscan(X_scaled, eps_values, min_samples_values)
    
    # Apply DBSCAN with best parameters
    if best_params is not None:
        dbscan, labels = apply_dbscan(X_scaled, best_params[0], best_params[1])
    
    print("\nDBSCAN clustering completed successfully!")
