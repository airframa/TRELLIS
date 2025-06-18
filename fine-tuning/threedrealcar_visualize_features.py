#!/usr/bin/env python3
"""
Visualize DINOv2 features and SLAT latents from TRELLIS
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

def load_dinov2_features(feature_path):
    """Load DINOv2 features from NPZ file"""
    data = np.load(feature_path)
    print(f"DINOv2 feature keys: {list(data.keys())}")
    
    features = data['patchtokens']  # Shape: (N, 1024)
    indices = data['indices']        # Shape: (N, 3) - voxel coordinates
    
    print(f"Feature shape: {features.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Number of voxels with features: {features.shape[0]}")
    
    return features, indices

def load_slat_latents(latent_path):
    """Load SLAT latents from NPZ file"""
    data = np.load(latent_path)
    print(f"SLAT latent keys: {list(data.keys())}")
    
    feats = data['feats']    # Shape: (N, latent_dim)
    coords = data['coords']  # Shape: (N, 3) - voxel coordinates
    
    print(f"Latent shape: {feats.shape}")
    print(f"Coords shape: {coords.shape}")
    print(f"Latent dimension: {feats.shape[1]}")
    print(f"Number of voxels with latents: {feats.shape[0]}")
    
    return feats, coords

def visualize_3d_voxels(coords, values=None, title="3D Voxel Visualization", 
                       save_path=None, resolution=64):
    """Visualize voxel coordinates in 3D with optional color values"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # If values provided, use them for coloring
    if values is not None:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                           c=values, cmap='viridis', s=50, alpha=0.6)
        plt.colorbar(scatter, ax=ax, pad=0.1)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                  c='blue', s=50, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set limits based on resolution
    ax.set_xlim(0, resolution)
    ax.set_ylim(0, resolution)
    ax.set_zlim(0, resolution)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def visualize_feature_statistics(features, feature_type="Features", save_path=None):
    """Visualize statistics of features"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature magnitude distribution
    feature_norms = np.linalg.norm(features, axis=1)
    axes[0, 0].hist(feature_norms, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title(f'{feature_type} Magnitude Distribution')
    axes[0, 0].set_xlabel('L2 Norm')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Mean activation per dimension
    mean_activations = np.mean(features, axis=0)
    axes[0, 1].plot(mean_activations, alpha=0.7)
    axes[0, 1].set_title(f'Mean Activation per {feature_type} Dimension')
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Mean Activation')
    
    # 3. Feature correlation matrix (sample if too large)
    sample_size = min(100, features.shape[1])
    sample_indices = np.random.choice(features.shape[1], sample_size, replace=False)
    feature_subset = features[:, sample_indices]
    correlation = np.corrcoef(feature_subset.T)
    
    im = axes[1, 0].imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[1, 0].set_title(f'{feature_type} Correlation Matrix (sampled)')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Dimension')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Activation sparsity
    sparsity = np.sum(np.abs(features) < 0.01, axis=1) / features.shape[1]
    axes[1, 1].hist(sparsity, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title(f'{feature_type} Sparsity Distribution')
    axes[1, 1].set_xlabel('Sparsity (fraction near zero)')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def visualize_feature_reduction(features, coords, method='pca', n_components=3, 
                               title="Feature Reduction", save_path=None):
    """Visualize features using dimensionality reduction"""
    print(f"Applying {method.upper()} to reduce {features.shape[1]}D to {n_components}D...")
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(features)
        print(f"Explained variance ratio: {reducer.explained_variance_ratio_}")
    elif method == 'tsne':
        # For t-SNE, first reduce with PCA if dimension is very high
        if features.shape[1] > 50:
            pca = PCA(n_components=50)
            features_pca = pca.fit_transform(features)
        else:
            features_pca = features
        
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(features_pca)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                           c=coords[:, 2], cmap='viridis', s=50, alpha=0.6)
        
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_zlabel(f'{method.upper()} 3')
        ax.set_title(f'{title} - {method.upper()} Visualization')
        
        plt.colorbar(scatter, ax=ax, label='Z coordinate')
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1],
                           c=coords[:, 2], cmap='viridis', s=50, alpha=0.6)
        
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_title(f'{title} - {method.upper()} Visualization')
        
        plt.colorbar(scatter, ax=ax, label='Z coordinate')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return reduced

def compare_features_and_latents(dinov2_features, dinov2_coords, 
                                slat_features, slat_coords,
                                save_dir=None):
    """Compare DINOv2 features with SLAT latents"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Dimensionality comparison
    labels = ['DINOv2', 'SLAT']
    dimensions = [dinov2_features.shape[1], slat_features.shape[1]]
    counts = [dinov2_features.shape[0], slat_features.shape[0]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, dimensions, width, label='Dimensions', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, counts, width, label='Voxel Count', color='green', alpha=0.7)
    axes[0, 0].set_xlabel('Feature Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Feature Dimensions and Voxel Counts')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].legend()
    
    # 2. Magnitude comparison
    dinov2_norms = np.linalg.norm(dinov2_features, axis=1)
    slat_norms = np.linalg.norm(slat_features, axis=1)
    
    axes[0, 1].hist(dinov2_norms, bins=50, alpha=0.5, label='DINOv2', color='blue', density=True)
    axes[0, 1].hist(slat_norms, bins=50, alpha=0.5, label='SLAT', color='red', density=True)
    axes[0, 1].set_xlabel('Feature Magnitude (L2 norm)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Feature Magnitude Distribution')
    axes[0, 1].legend()
    
    # 3. Spatial coverage comparison
    resolution = 64
    dinov2_coverage = np.zeros((resolution, resolution, resolution))
    slat_coverage = np.zeros((resolution, resolution, resolution))
    
    for coord in dinov2_coords:
        dinov2_coverage[coord[0], coord[1], coord[2]] = 1
    for coord in slat_coords:
        slat_coverage[coord[0], coord[1], coord[2]] = 1
    
    # Show XY projection
    dinov2_xy = np.max(dinov2_coverage, axis=2)
    slat_xy = np.max(slat_coverage, axis=2)
    
    axes[1, 0].imshow(dinov2_xy, cmap='Blues', alpha=0.7, origin='lower')
    axes[1, 0].set_title('DINOv2 Coverage (XY projection)')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    
    axes[1, 1].imshow(slat_xy, cmap='Reds', alpha=0.7, origin='lower')
    axes[1, 1].set_title('SLAT Coverage (XY projection)')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/feature_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Setup paths
    dataset_dir = Path("./assets/3drealcar/sample-data-processed-trellis")
    
    # Get SHA256 from metadata
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    sha256 = metadata.iloc[0]['sha256']
    print(f"Processing instance: {sha256}")
    
    # Create output directory for visualizations
    vis_dir = dataset_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Load DINOv2 features
    print("\n=== Loading DINOv2 Features ===")
    dinov2_path = dataset_dir / "features" / "dinov2_vitl14_reg" / f"{sha256}.npz"
    dinov2_features, dinov2_indices = load_dinov2_features(dinov2_path)
    
    # Load SLAT latents
    print("\n=== Loading SLAT Latents ===")
    slat_path = dataset_dir / "latents" / "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16" / f"{sha256}.npz"
    slat_features, slat_coords = load_slat_latents(slat_path)
    
    # Visualize DINOv2 features
    print("\n=== Visualizing DINOv2 Features ===")
    visualize_3d_voxels(dinov2_indices, 
                       title="DINOv2 Feature Voxel Distribution",
                       save_path=vis_dir / "dinov2_voxels.png")
    
    visualize_feature_statistics(dinov2_features, 
                               feature_type="DINOv2",
                               save_path=vis_dir / "dinov2_statistics.png")
    
    visualize_feature_reduction(dinov2_features, dinov2_indices, 
                              method='pca', n_components=3,
                              title="DINOv2 Features",
                              save_path=vis_dir / "dinov2_pca.png")
    
    # Visualize SLAT latents
    print("\n=== Visualizing SLAT Latents ===")
    visualize_3d_voxels(slat_coords,
                       title="SLAT Latent Voxel Distribution", 
                       save_path=vis_dir / "slat_voxels.png")
    
    visualize_feature_statistics(slat_features,
                               feature_type="SLAT",
                               save_path=vis_dir / "slat_statistics.png")
    
    visualize_feature_reduction(slat_features, slat_coords,
                              method='pca', n_components=3,
                              title="SLAT Latents",
                              save_path=vis_dir / "slat_pca.png")
    
    # Compare features and latents
    print("\n=== Comparing DINOv2 and SLAT ===")
    compare_features_and_latents(dinov2_features, dinov2_indices,
                               slat_features, slat_coords,
                               save_dir=vis_dir)
    
    print(f"\n✅ Visualizations saved to: {vis_dir}")

if __name__ == "__main__":
    main()