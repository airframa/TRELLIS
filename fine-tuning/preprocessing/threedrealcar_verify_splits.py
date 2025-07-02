#!/usr/bin/env python3
"""
Verify the train/val/test splits are correct and non-overlapping
"""
import os
import pandas as pd
from pathlib import Path
import json

OUTPUT_DIR = Path("/data/francMB/threedrealcar/splits")

def verify_splits():
    """Verify splits are correct"""
    
    # Load split info
    with open(OUTPUT_DIR / 'split_info.json', 'r') as f:
        split_info = json.load(f)
    
    print("Split Verification Report")
    print("=" * 50)
    
    # Check each split
    all_samples = set()
    for split_name in ['train', 'val', 'test']:
        split_dir = OUTPUT_DIR / f"3drealcar-{split_name}"
        
        print(f"\n{split_name.upper()} Split:")
        
        # Check metadata
        metadata_path = split_dir / "metadata.csv"
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            print(f"  Metadata entries: {len(df)}")
            
            # Check for duplicates
            duplicates = df['sha256'].duplicated().sum()
            print(f"  Duplicates in metadata: {duplicates}")
            
            # Check aesthetic score distribution
            print(f"  Aesthetic score range: {df['aesthetic_score'].min():.2f} - {df['aesthetic_score'].max():.2f}")
            print(f"  Mean aesthetic score: {df['aesthetic_score'].mean():.2f}")
            
            # Collect all samples
            split_samples = set(df['sha256'].tolist())
            
            # Check overlap with other splits
            overlap = all_samples.intersection(split_samples)
            if overlap:
                print(f"  ⚠️  WARNING: {len(overlap)} samples overlap with other splits!")
            else:
                print(f"  ✅ No overlap with other splits")
            
            all_samples.update(split_samples)
            
            # Verify symbolic links
            broken_links = 0
            total_links = 0
            missing_files = []
            
            # Check each file type
            for sha256 in df['sha256']:
                # Check features (features/dinov2_vitl14_reg/*.npz)
                feature_file = split_dir / 'features' / 'dinov2_vitl14_reg' / f"{sha256}.npz"
                if feature_file.exists():
                    total_links += 1
                    if not feature_file.resolve().exists():
                        broken_links += 1
                else:
                    missing_files.append(f"features/dinov2_vitl14_reg/{sha256}.npz")
                
                # Check latents (latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/*.npz)
                latent_file = split_dir / 'latents' / 'dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16' / f"{sha256}.npz"
                if latent_file.exists():
                    total_links += 1
                    if not latent_file.resolve().exists():
                        broken_links += 1
                else:
                    missing_files.append(f"latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/{sha256}.npz")
                
                # Check voxels (*.ply)
                voxel_file = split_dir / 'voxels' / f"{sha256}.ply"
                if voxel_file.exists():
                    total_links += 1
                    if not voxel_file.resolve().exists():
                        broken_links += 1
                else:
                    missing_files.append(f"voxels/{sha256}.ply")
                
                # Check renders (directory)
                render_dir = split_dir / 'renders' / sha256
                if render_dir.exists():
                    total_links += 1
                    if not render_dir.resolve().exists():
                        broken_links += 1
                else:
                    missing_files.append(f"renders/{sha256}/")
            
            print(f"  Total symbolic links: {total_links}")
            print(f"  Expected links: {len(df) * 4}")  # 4 items per sample
            print(f"  Broken links: {broken_links}")
            print(f"  Missing files: {len(missing_files)}")
            
            if missing_files and len(missing_files) < 10:
                print("  First few missing files:")
                for mf in missing_files[:5]:
                    print(f"    - {mf}")
        else:
            print(f"  ⚠️  No metadata.csv found!")
    
    # Summary
    print(f"\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Total unique samples across all splits: {len(all_samples)}")
    print(f"Expected total samples: {split_info['total_samples']}")
    
    if len(all_samples) == split_info['total_samples']:
        print("✅ Sample count matches expected!")
    else:
        print("⚠️  Sample count mismatch!")

if __name__ == "__main__":
    verify_splits()