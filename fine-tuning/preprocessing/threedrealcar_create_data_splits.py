#!/usr/bin/env python3
"""
Create train/val/test splits for 3DRealCar dataset using symbolic links
Preserves original data structure without duplication
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from collections import defaultdict
import json

# Configuration
BASE_DIR = Path("/data/francMB/threedrealcar")
OUTPUT_DIR = Path("/data/francMB/threedrealcar/splits")
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}
RANDOM_SEED = 42

# Folders to process (excluding pipeline-workspace)
FOLDERS = [
    "0-200", "200-400", "400-600", "600-800", "800-1000",
    "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2045",
    "HQ200", "HQ300-1", "HQ300-2"
]

def collect_all_samples(use_clean=False):
    """Collect all samples from all folders"""
    
    # Check if we should use pre-cleaned data
    clean_samples_path = OUTPUT_DIR / 'clean_samples.json'
    if use_clean and clean_samples_path.exists():
        print("Using pre-cleaned samples...")
        with open(clean_samples_path, 'r') as f:
            all_samples = json.load(f)
        
        # Create folder mapping
        folder_mapping = {s['sha256']: s['folder'] for s in all_samples}
        print(f"Loaded {len(all_samples)} clean samples")
        return all_samples, folder_mapping
    
    all_samples = []
    folder_mapping = {}
    seen_sha256 = set()  # Track unique samples
    
    for folder in FOLDERS:
        folder_path = BASE_DIR / folder
        metadata_path = folder_path / "metadata.csv"
        
        if not metadata_path.exists():
            print(f"Warning: {metadata_path} not found, skipping...")
            continue
            
        # Read metadata
        df = pd.read_csv(metadata_path)
        print(f"Found {len(df)} entries in {folder}")
        
        # Add folder information - skip duplicates
        unique_count = 0
        for idx, row in df.iterrows():
            sample_id = row['sha256']
            
            # Skip if we've seen this sample before
            if sample_id in seen_sha256:
                continue
                
            # Verify files exist
            features_exist = (folder_path / 'features' / 'dinov2_vitl14_reg' / f"{sample_id}.npz").exists()
            latents_exist = (folder_path / 'latents' / 'dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16' / f"{sample_id}.npz").exists()
            voxels_exist = (folder_path / 'voxels' / f"{sample_id}.ply").exists()
            renders_exist = (folder_path / 'renders' / sample_id).exists()
            
            if features_exist and latents_exist and voxels_exist and renders_exist:
                all_samples.append({
                    'sha256': sample_id,
                    'folder': folder,
                    'aesthetic_score': row['aesthetic_score'],
                    'num_voxels': row['num_voxels'],
                    'num_views': row['num_views']
                })
                folder_mapping[sample_id] = folder
                seen_sha256.add(sample_id)
                unique_count += 1
        
        print(f"  → Added {unique_count} unique samples with complete files")
    
    print(f"\nTotal unique samples collected: {len(all_samples)}")
    return all_samples, folder_mapping

def stratified_split(samples, split_ratios, random_seed=42):
    """
    Perform stratified split based on aesthetic score ranges
    This ensures similar quality distribution in train/val/test
    """
    np.random.seed(random_seed)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(samples)
    
    # Create aesthetic score bins for stratification
    df['score_bin'] = pd.qcut(df['aesthetic_score'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split
    train_samples = []
    val_samples = []
    test_samples = []
    
    # Stratified split by score bins
    for score_bin in df['score_bin'].unique():
        bin_samples = df[df['score_bin'] == score_bin]
        n_samples = len(bin_samples)
        
        n_train = int(n_samples * split_ratios['train'])
        n_val = int(n_samples * split_ratios['val'])
        n_test = n_samples - n_train - n_val
        
        train_samples.extend(bin_samples.iloc[:n_train].to_dict('records'))
        val_samples.extend(bin_samples.iloc[n_train:n_train+n_val].to_dict('records'))
        test_samples.extend(bin_samples.iloc[n_train+n_val:].to_dict('records'))
    
    # Shuffle final splits
    np.random.shuffle(train_samples)
    np.random.shuffle(val_samples)
    np.random.shuffle(test_samples)
    
    print(f"\nSplit results:")
    print(f"Train: {len(train_samples)} samples ({len(train_samples)/len(samples)*100:.1f}%)")
    print(f"Val: {len(val_samples)} samples ({len(val_samples)/len(samples)*100:.1f}%)")
    print(f"Test: {len(test_samples)} samples ({len(test_samples)/len(samples)*100:.1f}%)")
    
    return train_samples, val_samples, test_samples

def create_split_directories(output_dir, splits):
    """Create directory structure with symbolic links"""
    
    for split_name in ['train', 'val', 'test']:
        split_dir = output_dir / f"3drealcar-{split_name}"
        
        # Create clean directory
        if split_dir.exists():
            response = input(f"{split_dir} exists. Remove and recreate? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(split_dir)
            else:
                print(f"Skipping {split_name}...")
                continue
        
        split_dir.mkdir(parents=True)
        
        # Create subdirectories
        for subdir in ['features', 'latents', 'renders', 'voxels']:
            (split_dir / subdir).mkdir()
    
    print("\nCreated split directories")

def create_symbolic_links(output_dir, splits, folder_mapping):
    """Create symbolic links to original data"""
    
    split_names = ['train', 'val', 'test']
    
    for split_name, samples in zip(split_names, splits):
        split_dir = output_dir / f"3drealcar-{split_name}"
        
        print(f"\nCreating symbolic links for {split_name}...")
        
        # Track samples per folder for this split
        samples_by_folder = defaultdict(list)
        for sample in samples:
            samples_by_folder[sample['folder']].append(sample['sha256'])
        
        # Create links
        for sample in samples:
            sha256 = sample['sha256']
            source_folder = folder_mapping[sha256]
            source_base = BASE_DIR / source_folder
            
            # Link files/directories based on type
            # Features: features/dinov2_vitl14_reg/sha256.npz
            source_path = source_base / 'features' / 'dinov2_vitl14_reg' / f"{sha256}.npz"
            target_path = split_dir / 'features' / 'dinov2_vitl14_reg' / f"{sha256}.npz"
            # Create subdirectory if it doesn't exist
            target_path.parent.mkdir(exist_ok=True, parents=True)
            if source_path.exists():
                try:
                    target_path.symlink_to(source_path)
                except FileExistsError:
                    pass
            else:
                print(f"Warning: {source_path} not found")
            
            # Latents: latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/sha256.npz
            source_path = source_base / 'latents' / 'dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16' / f"{sha256}.npz"
            target_path = split_dir / 'latents' / 'dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16' / f"{sha256}.npz"
            # Create subdirectory if it doesn't exist
            target_path.parent.mkdir(exist_ok=True, parents=True)
            if source_path.exists():
                try:
                    target_path.symlink_to(source_path)
                except FileExistsError:
                    pass
            else:
                print(f"Warning: {source_path} not found")
            
            # Voxels: sha256.ply (in root of voxels/)
            source_path = source_base / 'voxels' / f"{sha256}.ply"
            target_path = split_dir / 'voxels' / f"{sha256}.ply"
            if source_path.exists():
                try:
                    target_path.symlink_to(source_path)
                except FileExistsError:
                    pass
            else:
                print(f"Warning: {source_path} not found")
            
            # Renders: sha256/ directory (in root of renders/)
            source_path = source_base / 'renders' / sha256
            target_path = split_dir / 'renders' / sha256
            if source_path.exists() and source_path.is_dir():
                try:
                    target_path.symlink_to(source_path)
                except FileExistsError:
                    pass
            else:
                print(f"Warning: {source_path} not found")
        
        # Create metadata CSV for this split
        create_split_metadata(split_dir, samples, folder_mapping)
        
        # Create latent CSV for this split
        create_split_latent_csv(split_dir, samples, folder_mapping)
        
        print(f"Created {len(samples)} symbolic links for {split_name}")

def create_split_metadata(split_dir, samples, folder_mapping):
    """Create metadata.csv for a split by combining from source folders"""
    
    all_metadata = []
    
    # Group samples by source folder
    samples_by_folder = defaultdict(list)
    for sample in samples:
        samples_by_folder[sample['folder']].append(sample['sha256'])
    
    # Read metadata from each source folder
    for folder, sample_ids in samples_by_folder.items():
        source_metadata = BASE_DIR / folder / "metadata.csv"
        if source_metadata.exists():
            df = pd.read_csv(source_metadata)
            # Filter to only include samples in this split
            df_filtered = df[df['sha256'].isin(sample_ids)]
            all_metadata.append(df_filtered)
    
    # Combine and save
    if all_metadata:
        combined_df = pd.concat(all_metadata, ignore_index=True)
        combined_df.to_csv(split_dir / "metadata.csv", index=False)
        print(f"Created metadata.csv with {len(combined_df)} entries")

def create_split_latent_csv(split_dir, samples, folder_mapping):
    """Create latent CSV for a split"""
    
    all_latents = []
    
    # Group samples by source folder
    samples_by_folder = defaultdict(list)
    for sample in samples:
        samples_by_folder[sample['folder']].append(sample['sha256'])
    
    # Read latent CSVs from each source folder
    for folder, sample_ids in samples_by_folder.items():
        source_latent = BASE_DIR / folder / "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16_0.csv"
        if source_latent.exists():
            df = pd.read_csv(source_latent)
            # Filter to only include samples in this split
            df_filtered = df[df['sha256'].isin(sample_ids)]
            all_latents.append(df_filtered)
    
    # Combine and save
    if all_latents:
        combined_df = pd.concat(all_latents, ignore_index=True)
        output_path = split_dir / "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16_0.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"Created latent CSV with {len(combined_df)} entries")

def save_split_info(output_dir, train_samples, val_samples, test_samples):
    """Save split information for reproducibility"""
    
    split_info = {
        'split_ratios': SPLIT_RATIOS,
        'random_seed': RANDOM_SEED,
        'total_samples': len(train_samples) + len(val_samples) + len(test_samples),
        'splits': {
            'train': {
                'count': len(train_samples),
                'samples': [s['sha256'] for s in train_samples]
            },
            'val': {
                'count': len(val_samples),
                'samples': [s['sha256'] for s in val_samples]
            },
            'test': {
                'count': len(test_samples),
                'samples': [s['sha256'] for s in test_samples]
            }
        },
        'folders_used': FOLDERS
    }
    
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nSaved split information to {output_dir / 'split_info.json'}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument('--clean', action='store_true', help='Use pre-cleaned samples')
    args = parser.parse_args()
    
    print("3DRealCar Dataset Train/Val/Test Split Creator")
    print("=" * 50)
    
    # Collect all samples
    print("\n1. Collecting samples from all folders...")
    all_samples, folder_mapping = collect_all_samples(use_clean=args.clean)
    
    # Perform stratified split
    print("\n2. Performing stratified split...")
    train_samples, val_samples, test_samples = stratified_split(
        all_samples, SPLIT_RATIOS, RANDOM_SEED
    )
    
    # Verify no overlaps
    train_ids = set(s['sha256'] for s in train_samples)
    val_ids = set(s['sha256'] for s in val_samples)
    test_ids = set(s['sha256'] for s in test_samples)
    
    assert len(train_ids & val_ids) == 0, "Train and Val overlap!"
    assert len(train_ids & test_ids) == 0, "Train and Test overlap!"
    assert len(val_ids & test_ids) == 0, "Val and Test overlap!"
    print("✅ Verified: No overlaps between splits")
    
    # Create directory structure
    print("\n3. Creating directory structure...")
    create_split_directories(OUTPUT_DIR, [train_samples, val_samples, test_samples])
    
    # Create symbolic links
    print("\n4. Creating symbolic links...")
    create_symbolic_links(
        OUTPUT_DIR, 
        [train_samples, val_samples, test_samples],
        folder_mapping
    )
    
    # Save split information
    print("\n5. Saving split information...")
    save_split_info(OUTPUT_DIR, train_samples, val_samples, test_samples)
    
    print("\n✅ Dataset split complete!")
    print(f"\nCreated splits in: {OUTPUT_DIR}")
    print("- 3drealcar-train/")
    print("- 3drealcar-val/")
    print("- 3drealcar-test/")

if __name__ == "__main__":
    main()