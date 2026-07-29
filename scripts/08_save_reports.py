# -*- coding: utf-8 -*-
"""
08_save_reports.py - Save all reports to CSV files
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("="*60)
print("SAVING REPORTS")
print("="*60)

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_path, "data", "segmentation_data.csv")
df = pd.read_csv(data_path)
df_clean = df.drop('ID', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

df_clustered = df_clean.copy()
df_clustered['Cluster'] = kmeans_labels

os.makedirs("../results/reports", exist_ok=True)

df_clustered.to_csv("../results/reports/clustered_data.csv", index=False)
print("Clustered data saved to: results/reports/clustered_data.csv")

cluster_profiles = df_clustered.groupby('Cluster').mean()
cluster_profiles['Size'] = df_clustered.groupby('Cluster').size()
cluster_profiles.to_csv("../results/reports/cluster_profiles.csv")
print("Cluster profiles saved to: results/reports/cluster_profiles.csv")

cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
cluster_counts_df = pd.DataFrame({
    'Cluster': cluster_counts.index,
    'Count': cluster_counts.values,
    'Percentage': (cluster_counts.values / len(df_clustered) * 100).round(1)
})
cluster_counts_df.to_csv("../results/reports/cluster_counts.csv", index=False)
print("Cluster counts saved to: results/reports/cluster_counts.csv")

stats = df_clustered.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])
stats.to_csv("../results/reports/cluster_statistics.csv")
print("Cluster statistics saved to: results/reports/cluster_statistics.csv")

print("\n" + "="*60)
print("[All reports saved successfully!")
print("="*60)
print("\nReports saved in: results/reports/")
print("  - clustered_data.csv")
print("  - cluster_profiles.csv")
print("  - cluster_counts.csv")
print("  - cluster_statistics.csv")