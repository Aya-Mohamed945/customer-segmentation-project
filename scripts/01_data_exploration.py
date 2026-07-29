
"""
Module: Data Exploration
Purpose: Load and explore the dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load the dataset from CSV file"""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def explore_data(df):
    """Perform initial data exploration"""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData Info:")
    print(df.info())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

def plot_categorical_distribution(df, save_path=None):
    """Plot distribution of categorical variables"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution of Categorical Variables', fontsize=16, fontweight='bold')
    
    categorical_cols = ['Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size']
    labels_dict = {
        'Sex': ['Male', 'Female'],
        'Marital status': ['Single', 'Non-single'],
        'Education': ['Other', 'High School', 'University', 'Graduate'],
        'Occupation': ['Unemployed', 'Skilled', 'Management'],
        'Settlement size': ['Small', 'Mid-sized', 'Big']
    }
    
    for idx, col in enumerate(categorical_cols):
        row = idx // 3
        col_idx = idx % 3
        ax = axes[row, col_idx]
        
        counts = df[col].value_counts().sort_index()
        colors = sns.color_palette("husl", len(counts))
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.8)
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        if col in labels_dict:
            ax.set_xticks(range(len(labels_dict[col])))
            ax.set_xticklabels(labels_dict[col], rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_age_income_distribution(df, save_path=None):
    """Plot age and income distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Age and Income Distributions', fontsize=14, fontweight='bold')
    
    # Age distribution
    axes[0].hist(df['Age'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {df["Age"].mean():.1f}')
    axes[0].axvline(df['Age'].median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {df["Age"].median():.1f}')
    axes[0].set_title('Age Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Age (years)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Income distribution
    axes[1].hist(df['Income'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[1].axvline(df['Income'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: ${df["Income"].mean():,.0f}')
    axes[1].axvline(df['Income'].median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: ${df["Income"].median():,.0f}')
    axes[1].set_title('Income Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Income ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_correlation_matrix(df, save_path=None):
    """Plot correlation matrix"""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Variables', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("../data/segmentation_data.csv")
    df_clean = df.drop('ID', axis=1)
    
    # Explore data
    explore_data(df_clean)
    
    # Generate visualizations
    plot_categorical_distribution(df_clean, "../results/plots/categorical_distribution.png")
    plot_age_income_distribution(df_clean, "../results/plots/age_income_distribution.png")
    plot_correlation_matrix(df_clean, "../results/plots/correlation_matrix.png")
    
    print("\nData exploration completed successfully!")