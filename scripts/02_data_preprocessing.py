"""
Module: Data Preprocessing
Purpose: Standardize data and perform PCA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(df):
    """Preprocess data: drop ID and standardize"""
    # Drop ID column if exists
    if 'ID' in df.columns:
        df_clean = df.drop('ID', axis=1)
    else:
        df_clean = df.copy()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df_clean.columns)
    
    print("Data preprocessing completed!")
    print(f"Shape after preprocessing: {X_scaled_df.shape}")
    
    return df_clean, X_scaled, X_scaled_df, scaler

def perform_pca(X_scaled):
    """Perform PCA and return transformed data"""
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\n" + "="*60)
    print("PCA VARIANCE EXPLAINED")
    print("="*60)
    for i in range(min(7, len(explained_variance))):
        print(f"PC{i+1}: {explained_variance[i]*100:.2f}% (Cumulative: {cumulative_variance[i]*100:.2f}%)")
    
    # Determine optimal components
    n_components = sum(1 for v in cumulative_variance if v < 0.90) + 1
    print(f"\nRecommended components: {n_components} (covers {cumulative_variance[n_components-1]*100:.2f}% variance)")
    
    return pca, X_pca

def plot_pca_variance(pca, save_path=None):
    """Plot PCA variance explained"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PCA Variance Analysis', fontsize=14, fontweight='bold')
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Individual variance
    axes[0].bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, color='skyblue')
    axes[0].plot(range(1, len(explained_variance)+1), cumulative_variance, 'ro-', linewidth=2)
    axes[0].set_xlabel('Principal Components')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Individual Variance Explained')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative variance
    axes[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'bo-', linewidth=2)
    axes[1].axhline(y=0.85, color='r', linestyle='--', linewidth=2, label='85% Variance')
    axes[1].axhline(y=0.90, color='g', linestyle='--', linewidth=2, label='90% Variance')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("../data/segmentation_data.csv")
    
    # Preprocess
    df_clean, X_scaled, X_scaled_df, scaler = preprocess_data(df)
    
    # Perform PCA
    pca, X_pca = perform_pca(X_scaled)
    plot_pca_variance(pca, "../results/plots/pca_variance.png")
    
    print("\nPreprocessing completed successfully!")
