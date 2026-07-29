# -*- coding: utf-8 -*-
"""
Module: Visualization
Purpose: Visualize clustering results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_path, "data", "segmentation_data.csv")
df = pd.read_csv(data_path)
df_clean = df.drop('ID', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_centroids = kmeans.cluster_centers_

hierarchical = AgglomerativeClustering(n_clusters=9, metric='euclidean', linkage='ward')
hier_labels = hierarchical.fit_predict(X_scaled)

pca_vis = PCA(n_components=2)
X_pca_vis = pca_vis.fit_transform(X_scaled)
centroids_pca = pca_vis.transform(kmeans_centroids)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1],
                     c=kmeans_labels, cmap='viridis', alpha=0.6, s=30)

plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidths=2,
           label='Cluster Centers', zorder=5)

plt.title('K-Means Clustering Results (k=9)', fontsize=14, fontweight='bold')
plt.xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter, label='Cluster Number')
plt.legend()
plt.grid(True, alpha=0.3)

# حفظ الصورة
os.makedirs("../results/plots", exist_ok=True)
plt.savefig("../results/plots/kmeans_clusters.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved to: results/plots/kmeans_clusters.png")