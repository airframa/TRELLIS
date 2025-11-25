#!/usr/bin/env python3
"""
Merge all processed batches into a single unified dataset.
Creates symlinks to avoid data duplication.
Generates train/val/test splits.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import shutil
import numpy as np

# Split ratios
SPLIT_RATIOS = {
    "train": 0.95,
    "val": 0.0,
    "test": 0.05
}

def create_symlink_relative(src, dst):
    """Create a relative symlink."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    
    # Create relative path
    rel_path = os.path.relpath(src, dst.parent)
    os.symlink(rel_path, dst)

def merge_datasets(input_base, output_dir):
    """Merge all batch datasets into one."""
    input_base = Path(input_base)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Scanning for batch datasets...")
    
    # Find all batch directories with metadata
    batch_dirs = []
    for d in input_base.iterdir():
        if d.is_dir() and (d / "metadata.csv").exists():
            batch_dirs.append(d)
    
    batch_dirs.sort(key=lambda x: x.name)
    print(f"Found {len(batch_dirs)} batch datasets")
    
    # Load all metadata
    all_metadata = []
    for batch_dir in batch_dirs:
        print(f"  Loading {batch_dir.name}...")
        df = pd.read_csv(batch_dir / "metadata.csv")
        df['batch'] = batch_dir.name
        all_metadata.append(df)
    
    # Combine metadata
    merged_df = pd.concat(all_metadata, ignore_index=True)
    print(f"\nTotal samples: {len(merged_df)}")
    
    # Create directory structure
    (output_dir / "renders").mkdir(exist_ok=True)
    (output_dir / "voxels").mkdir(exist_ok=True)
    (output_dir / "features" / "dinov2_vitl14_reg").mkdir(parents=True, exist_ok=True)
    
    # Create symlinks
    print("\nCreating symlinks...")
    for _, row in merged_df.iterrows():
        sha256 = row['sha256']
        batch = row['batch']
        batch_dir = input_base / batch
        
        # Symlink renders directory
        src_renders = batch_dir / "renders" / sha256
        dst_renders = output_dir / "renders" / sha256
        if src_renders.exists():
            create_symlink_relative(src_renders, dst_renders)
        
        # Symlink voxels
        src_voxel = batch_dir / "voxels" / f"{sha256}.ply"
        dst_voxel = output_dir / "voxels" / f"{sha256}.ply"
        if src_voxel.exists():
            create_symlink_relative(src_voxel, dst_voxel)
        
        # Symlink features
        src_feat = batch_dir / "features" / "dinov2_vitl14_reg" / f"{sha256}.npz"
        dst_feat = output_dir / "features" / "dinov2_vitl14_reg" / f"{sha256}.npz"
        if src_feat.exists():
            create_symlink_relative(src_feat, dst_feat)
    
    # Remove batch column and save metadata
    merged_df = merged_df.drop(columns=['batch'])
    merged_df.to_csv(output_dir / "metadata.csv", index=False)
    
    # Create instances.txt
    with open(output_dir / "instances.txt", 'w') as f:
        for sha256 in merged_df['sha256']:
            f.write(f"{sha256}\n")
    
    # Generate statistics
    stats = {
        'total_samples': len(merged_df),
        'rendered': merged_df['rendered'].sum(),
        'voxelized': merged_df['voxelized'].sum(),
        'with_features': merged_df['feature_dinov2_vitl14_reg'].sum() if 'feature_dinov2_vitl14_reg' in merged_df.columns else 0,
        'batches': len(batch_dirs),
        'batch_names': [d.name for d in batch_dirs]
    }
    
    with open(output_dir / "statistics.txt", 'w') as f:
        f.write("Merged 3DRealCar Dataset Statistics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total samples: {stats['total_samples']}\n")
        f.write(f"Rendered: {stats['rendered']}\n")
        f.write(f"Voxelized: {stats['voxelized']}\n")
        f.write(f"With DINOv2 features: {stats['with_features']}\n")
        f.write(f"Number of batches: {stats['batches']}\n")
        f.write("\nBatches:\n")
        for batch in stats['batch_names']:
            f.write(f"  - {batch}\n")
    
    print(f"\n{'='*60}")
    print("Merge completed!")
    print(f"Output: {output_dir}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_base", required=True,
                       help="Base directory containing processed batches")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for merged dataset")
    args = parser.parse_args()
    
    merge_datasets(args.input_base, args.output_dir)

if __name__ == "__main__":
    main()