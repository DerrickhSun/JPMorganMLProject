import helpers
import preprocessing
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def perform_pca(df_scaled, n_components=3, exclude_columns=None):
    """
    Perform PCA using PyTorch SVD, excluding specified columns.
    
    Args:
        df_scaled (pandas.DataFrame): Scaled data DataFrame
        n_components (int): Number of principal components to return
        exclude_columns (list): List of column names to exclude from PCA
        
    Returns:
        tuple: (U_reduced, S_reduced, Vh_reduced, explained_variance_ratio, feature_names)
    """
    # Remove excluded columns for PCA computation
    if exclude_columns is None:
        exclude_columns = []
    
    df_pca = df_scaled.drop(columns=exclude_columns)
    feature_names = df_pca.columns.tolist()
    
    # Convert DataFrame to PyTorch tensor
    data_tensor = torch.tensor(df_pca.values, dtype=torch.float32)
    
    # Center the data (subtract mean)
    data_centered = data_tensor - data_tensor.mean(dim=0)
    
    # Perform SVD
    U, S, Vh = torch.linalg.svd(data_centered, full_matrices=False)
    
    # Project data onto principal components
    # U contains the left singular vectors (principal components)
    # S contains the singular values (explained variance)
    # Vh contains the right singular vectors (feature loadings)
    
    # Take first n_components
    U_reduced = U[:, :n_components]
    S_reduced = S[:n_components]
    Vh_reduced = Vh[:n_components, :]
    
    # Calculate explained variance ratio
    explained_variance_ratio = (S_reduced ** 2) / (S ** 2).sum()
    
    return U_reduced, S_reduced, Vh_reduced, explained_variance_ratio, feature_names

def plot_pca_results(U_reduced, explained_variance_ratio, labels, n_components=3):
    """
    Plot the first 2-3 dimensions of the PCA results with colored labels.
    
    Args:
        U_reduced (torch.Tensor): Reduced principal components
        explained_variance_ratio (torch.Tensor): Explained variance ratios
        labels (pandas.Series): Label column for coloring
        n_components (int): Number of components
    """
    # Convert labels to numeric for coloring if they're categorical
    if labels.dtype == 'object':
        unique_labels = labels.unique()
        label_map = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = labels.map(label_map)
        cmap = plt.cm.tab10
    else:
        numeric_labels = labels
        cmap = plt.cm.viridis
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 2D projection (first two components)
    if n_components >= 2:
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(U_reduced[:, 0], U_reduced[:, 1], 
                            c=numeric_labels, alpha=0.6, s=1, cmap=cmap)
        plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
        plt.title('PCA: First Two Components (Colored by Label)')
        plt.grid(True, alpha=0.3)
        
        # Add colorbar if labels are categorical
        if labels.dtype == 'object':
            cbar = plt.colorbar(scatter)
            cbar.set_ticks(range(len(unique_labels)))
            cbar.set_ticklabels(unique_labels)
    
    # Plot 3D projection (first three components)
    if n_components >= 3:
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        scatter = ax.scatter(U_reduced[:, 0], U_reduced[:, 1], U_reduced[:, 2], 
                           c=numeric_labels, alpha=0.6, s=1, cmap=cmap)
        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
        ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]:.1%} variance)')
        ax.set_title('PCA: First Three Components (Colored by Label)')
    
    # Plot explained variance
    plt.subplot(1, 3, 3)
    plt.bar(range(1, n_components + 1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run PCA analysis.
    """
    print("Loading census data...")
    df_scaled = pd.read_csv("data/census-bureau-scaled.csv")
    exclude_columns = ['label']
    labels = df_scaled['label']
    
    print("\nPerforming PCA with PyTorch...")
    n_components = 3
    U_reduced, S_reduced, Vh_reduced, explained_variance_ratio, feature_names = perform_pca(
        df_scaled, n_components, exclude_columns
    )
    
    print(f"\nExplained variance ratio (first {n_components} components):")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {ratio:.1%}")
    
    print(f"\nCumulative explained variance: {explained_variance_ratio.sum():.1%}")
    
    print("\nPlotting results...")
    plot_pca_results(U_reduced, explained_variance_ratio, labels, n_components)
    
    
    # Show feature importance scores
    df_pca = df_scaled.drop(columns=exclude_columns)
    print("\nFeature importance scores (based on variance):")
    feature_importance = preprocessing.get_feature_importance_scores(df_pca)
    for i, (feature, score) in enumerate(list(feature_importance.items())[:10]):
        print(f"{i+1:2d}. {feature}: {score:.4f}")

if __name__ == "__main__":
    main()